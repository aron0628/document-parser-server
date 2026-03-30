"""OpenAI Vision + Chat API 클라이언트 (LangChain 기반)"""

import base64
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from app.pipeline.external.llm_utils import load_chat_model_with_retry

logger = logging.getLogger(__name__)

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

# 프로바이더별 API 키 kwarg 매핑
_PROVIDER_API_KEY_KWARG = {
    "openai": "api_key",
    "google_genai": "google_api_key",
    "gemini": "google_api_key",
    "xai": "xai_api_key",
    "grok": "xai_api_key",
}


def _resolve_api_key_kwarg(provider: str, api_key: str) -> dict:
    """프로바이더에 맞는 API 키 kwarg dict 반환."""
    key_name = _PROVIDER_API_KEY_KWARG.get(provider, "api_key")
    return {key_name: api_key}


def _is_rate_limit_error(e: Exception) -> bool:
    """LangChain이 래핑한 429 에러 감지"""
    err_name = type(e).__name__
    if "RateLimit" in err_name or "ResourceExhausted" in err_name:
        return True
    if hasattr(e, "status_code") and getattr(e, "status_code", 0) == 429:
        return True
    return False


async def describe_image(
    image_path: str,
    model_str: str,
    api_key: str,
    language: str = "Korean",
    context: Optional[str] = None,
    on_rate_limit=None,
) -> str:
    """이미지에 대한 설명 생성 (Vision API).

    load_chat_model로 프로바이더별 모델을 자동 초기화하며,
    on_rate_limit 콜백으로 글로벌 백프레셔를 유지한다.
    """
    provider = model_str.split("/", 1)[0] if "/" in model_str else "openai"
    api_key_kwarg = _resolve_api_key_kwarg(provider, api_key)

    model = load_chat_model_with_retry(model_str, timeout=120, **api_key_kwarg)

    # 멀티모달 메시지 구성
    image_data = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_data).decode("utf-8")
    b64_url = f"data:image/png;base64,{b64}"

    content_blocks: list[Any] = []

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
    content_blocks.append({"type": "image_url", "image_url": {"url": b64_url}})

    message = HumanMessage(content=content_blocks)

    try:
        response = await model.ainvoke([message])
        return response.content
    except Exception as e:
        if _is_rate_limit_error(e) and on_rate_limit:
            wait = 5.0
            await on_rate_limit(wait)
        raise


def _build_table_messages(
    table_content: str,
    language: str,
    image_path: Optional[str] = None,
) -> HumanMessage:
    """테이블 변환용 HumanMessage 생성.

    분기 기준:
    - 텍스트+이미지: 멀티모달 content 리스트 구성
    - 이미지만: 이미지 전용 프롬프트로 content 리스트 구성
    - 텍스트만: 기존 문자열 content 사용
    """
    if image_path:
        image_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")
        b64_url = f"data:image/png;base64,{b64}"
        image_part = {"type": "image_url", "image_url": {"url": b64_url}}

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

    return HumanMessage(content=content)


async def extract_table(
    table_content: str,
    model_str: str,
    api_key: str,
    language: str = "Korean",
    image_path: Optional[str] = None,
    on_rate_limit=None,
) -> str:
    """테이블 내용을 구조화된 마크다운 테이블로 변환.

    image_path 제공 시 Vision API를 통해 멀티모달 분석 수행.
    load_chat_model로 프로바이더별 모델을 자동 초기화한다.
    """
    provider = model_str.split("/", 1)[0] if "/" in model_str else "openai"
    api_key_kwarg = _resolve_api_key_kwarg(provider, api_key)

    model = load_chat_model_with_retry(model_str, timeout=120, **api_key_kwarg)

    message = _build_table_messages(table_content, language, image_path)

    try:
        response = await model.ainvoke([message])
        return response.content
    except Exception as e:
        if _is_rate_limit_error(e) and on_rate_limit:
            wait = 5.0
            await on_rate_limit(wait)
        raise
