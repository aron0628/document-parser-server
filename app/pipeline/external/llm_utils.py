"""LLM 모델 초기화 유틸리티 (react-agent 패턴)

사용법:
    from app.pipeline.external.llm_utils import load_chat_model, load_chat_model_with_retry

    model = load_chat_model("openai/gpt-4o", api_key="sk-xxx")
    model_with_retry = load_chat_model_with_retry("google_genai/gemini-2.5-flash", google_api_key="xxx")
"""

import logging
from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# LangChain이 인식하지 못하는 프로바이더 alias를 정규화
_PROVIDER_ALIASES = {
    "gemini": "google_genai",
    "grok": "xai",
}


def _normalize_provider(provider: str) -> str:
    """프로바이더 alias를 LangChain이 인식하는 이름으로 정규화."""
    return _PROVIDER_ALIASES.get(provider, provider)


def load_chat_model(fully_specified_name: str, **kwargs: Any) -> BaseChatModel:
    """모델 문자열로 LangChain Chat 모델 인스턴스 생성.

    Args:
        fully_specified_name: "provider/model" 형식 문자열 (예: "openai/gpt-4o")
        **kwargs: init_chat_model에 전달할 추가 인자 (api_key, temperature 등)
    """
    if "/" not in fully_specified_name:
        raise ValueError(
            f"모델 문자열에 프로바이더 접두사가 없습니다: '{fully_specified_name}'. "
            f"'provider/model-name' 형식으로 입력하세요."
        )
    provider, model = fully_specified_name.split("/", maxsplit=1)
    provider = _normalize_provider(provider)
    return cast(
        BaseChatModel, init_chat_model(model, model_provider=provider, **kwargs)
    )


def load_chat_model_with_retry(
    fully_specified_name: str,
    max_retries: int = 3,
    **kwargs: Any,
) -> BaseChatModel:
    """재시도 기능이 포함된 LangChain Chat 모델 인스턴스 생성.

    with_retry()로 429 Rate Limit, 5xx 서버 에러, 네트워크 에러에 대해
    지수 백오프 재시도를 적용한다.

    Args:
        fully_specified_name: "provider/model" 형식 문자열
        max_retries: 최대 재시도 횟수 (기본: 3)
        **kwargs: init_chat_model에 전달할 추가 인자
    """
    model = load_chat_model(fully_specified_name, **kwargs)
    return model.with_retry(
        stop_after_attempt=max_retries + 1,
    )
