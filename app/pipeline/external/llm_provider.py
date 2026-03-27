"""LLM 멀티 프로바이더 설정 및 유틸리티"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    """프로바이더별 설정"""

    base_url: str
    key_env: str           # 환경 변수명 (예: "OPENAI_API_KEY")
    key_field: str         # configurable dict의 키명 (예: "openai_api_key")
    supports_detail: bool  # Vision API의 detail 파라미터 지원 여부


_GEMINI_CONFIG = ProviderConfig(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    key_env="GOOGLE_API_KEY",
    key_field="google_api_key",
    supports_detail=False,
)

_XAI_CONFIG = ProviderConfig(
    base_url="https://api.x.ai/v1",
    key_env="XAI_API_KEY",
    key_field="xai_api_key",
    supports_detail=False,
)

PROVIDER_CONFIG = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        key_field="openai_api_key",
        supports_detail=True,
    ),
    "gemini": _GEMINI_CONFIG,
    "google_genai": _GEMINI_CONFIG,  # 관리자 UI 호환 alias
    "grok": _XAI_CONFIG,
    "xai": _XAI_CONFIG,              # 관리자 UI 호환 alias
}

# 관리자 UI(file-manager-admin)의 AVAILABLE_MODELS와 동일한 모델 목록
ALLOWED_VISION_MODELS = {
    # OpenAI
    "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    "openai/gpt-4.1-mini", "openai/gpt-4.1-nano",
    "openai/gpt-4o", "openai/gpt-4.1",
    # Google (google_genai prefix — 관리자 UI 호환)
    "google_genai/gemini-3.1-pro-preview",
    "google_genai/gemini-3.1-flash-lite-preview",
    "google_genai/gemini-3-flash-preview",
    # Google (gemini prefix — 레거시 호환)
    "gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro",
    # xAI (xai prefix — 관리자 UI 호환)
    "xai/grok-4.20-0309-reasoning", "xai/grok-4.20-0309-non-reasoning",
    "xai/grok-4.20-multi-agent-0309",
    "xai/grok-4-1-fast-reasoning", "xai/grok-4-1-fast-non-reasoning",
    # xAI (grok prefix — 레거시 호환)
    "grok/grok-3", "grok/grok-2-vision",
}

ALLOWED_TEXT_MODELS = {
    # OpenAI
    "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    "openai/gpt-4.1-mini", "openai/gpt-4.1-nano",
    "openai/gpt-4o", "openai/gpt-4.1",
    # Google (google_genai prefix — 관리자 UI 호환)
    "google_genai/gemini-3.1-pro-preview",
    "google_genai/gemini-3.1-flash-lite-preview",
    "google_genai/gemini-3-flash-preview",
    # Google (gemini prefix — 레거시 호환)
    "gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro",
    # xAI (xai prefix — 관리자 UI 호환)
    "xai/grok-4.20-0309-reasoning", "xai/grok-4.20-0309-non-reasoning",
    "xai/grok-4.20-multi-agent-0309",
    "xai/grok-4-1-fast-reasoning", "xai/grok-4-1-fast-non-reasoning",
    # xAI (grok prefix — 레거시 호환)
    "grok/grok-3",
}


def parse_model_string(model_str: str) -> tuple[str, str]:
    """모델 문자열을 (프로바이더, 모델명) 튜플로 파싱.

    예: "openai/gpt-4o" -> ("openai", "gpt-4o")
    슬래시가 없으면 ValueError 발생.
    """
    if "/" not in model_str:
        raise ValueError(
            f"모델 문자열에 프로바이더 접두사가 없습니다: '{model_str}'. "
            f"'provider/model-name' 형식으로 입력하세요."
        )
    provider, model_name = model_str.split("/", 1)
    return provider, model_name


def get_provider_config(provider: str) -> ProviderConfig:
    """프로바이더 설정 조회. 미등록 프로바이더면 ValueError 발생."""
    if provider not in PROVIDER_CONFIG:
        raise ValueError(
            f"지원하지 않는 프로바이더: '{provider}'. "
            f"지원 목록: {list(PROVIDER_CONFIG.keys())}"
        )
    return PROVIDER_CONFIG[provider]


def resolve_api_key(provider: str, keys: dict) -> str:
    """프로바이더에 맞는 API 키를 keys dict에서 조회.

    ProviderConfig.key_field로 keys dict에서 키를 선택하며,
    빈 문자열이면 ValueError 발생.
    """
    config = get_provider_config(provider)
    api_key = keys.get(config.key_field, "")
    if not api_key:
        raise ValueError(
            f"프로바이더 '{provider}'의 API 키가 없습니다. "
            f"'{config.key_field}' 키 또는 환경변수 '{config.key_env}'를 설정하세요."
        )
    return api_key


def validate_model(model_str: str, allowed: set) -> None:
    """모델이 허용 목록에 있는지 검증. 미허용 시 ValueError 발생."""
    if model_str not in allowed:
        raise ValueError(
            f"허용되지 않는 모델: '{model_str}'. "
            f"허용 목록: {sorted(allowed)}"
        )


def build_vision_image_block(b64_url: str, provider: str) -> dict:
    """Vision API용 이미지 content block 생성.

    OpenAI는 detail 파라미터를 포함하고, 나머지 프로바이더는 제외.
    """
    config = get_provider_config(provider)
    image_url: dict = {"url": b64_url}
    if config.supports_detail:
        image_url["detail"] = "low"
    return {"type": "image_url", "image_url": image_url}
