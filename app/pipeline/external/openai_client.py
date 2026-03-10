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
) -> str:
    """이미지에 대한 설명 생성 (Vision API)"""
    image_data = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_data).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"이 이미지의 내용을 {language}로 상세하게 설명해주세요. "
                            "차트, 다이어그램, 사진 등 어떤 유형인지 먼저 밝히고 핵심 내용을 설명해주세요.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ]

    result = await _call_openai(api_key, messages)
    return result["choices"][0]["message"]["content"]


async def extract_table(
    table_content: str,
    api_key: str,
    language: str = "Korean",
) -> str:
    """테이블 내용을 구조화된 마크다운 테이블로 변환"""
    messages = [
        {
            "role": "user",
            "content": (
                f"다음 테이블 데이터를 깔끔한 마크다운 테이블 형식으로 변환해주세요. "
                f"언어: {language}\n\n"
                f"테이블 데이터:\n{table_content}"
            ),
        }
    ]

    result = await _call_openai(api_key, messages)
    return result["choices"][0]["message"]["content"]
