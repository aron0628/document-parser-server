"""OpenAI Vision + Chat API 클라이언트 (httpx.AsyncClient)"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0

_TABLE_TEXT_PROMPT = (
    "다음 테이블 데이터를 깔끔한 마크다운 테이블 형식으로 변환해주세요. "
    "언어: {language}\n\n"
    "테이블 데이터:\n{table_content}"
)
_TABLE_MULTIMODAL_PROMPT = (
    "테이블 이미지와 추출 데이터를 참고하여 정확한 마크다운 테이블로 변환해주세요. "
    "이미지의 실제 내용을 우선시하세요. "
    "언어: {language}\n\n"
    "추출 데이터:\n{table_content}"
)
_TABLE_IMAGE_ONLY_PROMPT = (
    "테이블 이미지를 분석하여 정확한 마크다운 테이블로 변환해주세요. "
    "언어: {language}"
)


async def _call_openai(
    api_key: str,
    messages: list,
    model: str = "gpt-4o",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """OpenAI Chat API 호출 (retry 포함)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "max_tokens": 4096}

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(OPENAI_API_URL, headers=headers, json=payload)

                if response.status_code == 429:
                    wait = INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(f"OpenAI rate limit, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"OpenAI server error, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise
        except httpx.RequestError as e:
            if attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"OpenAI request error: {e}, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise

    raise RuntimeError(f"OpenAI API 호출 실패: {MAX_RETRIES}회 재시도 후에도 실패")


async def describe_image(
    image_path: str,
    api_key: str,
    language: str = "Korean",
    context: Optional[str] = None,
) -> str:
    """이미지에 대한 설명 생성 (Vision API)"""
    image_data = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_data).decode("utf-8")

    content_blocks = []

    if context:
        content_blocks.append({
            "type": "text",
            "text": f"다음은 이 이미지가 포함된 문서 페이지의 텍스트입니다:\n{context}"
        })

    if context:
        prompt = (f"위 문서 텍스트의 맥락을 참고하여, "
                  f"이 이미지의 내용을 {language}로 상세하게 설명해주세요. "
                  f"차트, 다이어그램, 사진 등 어떤 유형인지 먼저 밝히고 핵심 내용을 설명해주세요.")
    else:
        prompt = (f"이 이미지의 내용을 {language}로 상세하게 설명해주세요. "
                  f"차트, 다이어그램, 사진 등 어떤 유형인지 먼저 밝히고 핵심 내용을 설명해주세요.")

    content_blocks.append({"type": "text", "text": prompt})
    content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    messages = [{"role": "user", "content": content_blocks}]

    result = await _call_openai(api_key, messages)
    return result["choices"][0]["message"]["content"]


def _build_table_messages(
    table_content: str,
    language: str,
    image_path: Optional[str] = None,
) -> list:
    """테이블 변환용 messages 리스트 생성.

    분기 기준:
    - 텍스트+이미지: 멀티모달 content 리스트 구성
    - 이미지만: 이미지 전용 프롬프트로 content 리스트 구성
    - 텍스트만: 기존 문자열 content 사용
    """
    if image_path:
        image_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        }

        if table_content:
            text = _TABLE_MULTIMODAL_PROMPT.format(
                language=language, table_content=table_content
            )
        else:
            text = _TABLE_IMAGE_ONLY_PROMPT.format(language=language)

        content: Any = [{"type": "text", "text": text}, image_part]
    else:
        content = _TABLE_TEXT_PROMPT.format(
            language=language, table_content=table_content
        )

    return [{"role": "user", "content": content}]


async def extract_table(
    table_content: str,
    api_key: str,
    language: str = "Korean",
    image_path: Optional[str] = None,
) -> str:
    """테이블 내용을 구조화된 마크다운 테이블로 변환.

    image_path 제공 시 Vision API를 통해 멀티모달 분석 수행.
    하위 호환: image_path 없이 기존 호출 방식 그대로 사용 가능.
    """
    messages = _build_table_messages(table_content, language, image_path)
    result = await _call_openai(api_key, messages)
    return result["choices"][0]["message"]["content"]
