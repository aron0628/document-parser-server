"""LLM 멀티 프로바이더 설정 및 유틸리티"""

# 프로바이더별 (settings 키명, 환경변수명) 매핑
_PROVIDER_KEY_MAP: dict[str, tuple[str, str]] = {
    "openai":      ("openai_api_key",  "OPENAI_API_KEY"),
    "google_genai": ("google_api_key", "GOOGLE_API_KEY"),
    "gemini":      ("google_api_key",  "GOOGLE_API_KEY"),
    "xai":         ("xai_api_key",     "XAI_API_KEY"),
    "grok":        ("xai_api_key",     "XAI_API_KEY"),
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


def resolve_api_key(provider: str, keys: dict) -> str:
    """프로바이더에 맞는 API 키를 keys dict에서 조회.

    _PROVIDER_KEY_MAP에서 key_field를 선택하며,
    빈 문자열이면 ValueError 발생.
    """
    if provider not in _PROVIDER_KEY_MAP:
        raise ValueError(
            f"지원하지 않는 프로바이더: '{provider}'. "
            f"지원 목록: {list(_PROVIDER_KEY_MAP.keys())}"
        )
    key_field, key_env = _PROVIDER_KEY_MAP[provider]
    api_key = keys.get(key_field, "")
    if not api_key:
        raise ValueError(
            f"프로바이더 '{provider}'의 API 키가 없습니다. "
            f"'{key_field}' 키 또는 환경변수 '{key_env}'를 설정하세요."
        )
    return api_key


def validate_model(model_str: str, allowed: set) -> None:
    """모델이 허용 목록에 있는지 검증. 미허용 시 ValueError 발생."""
    if model_str not in allowed:
        raise ValueError(
            f"허용되지 않는 모델: '{model_str}'. "
            f"허용 목록: {sorted(allowed)}"
        )
