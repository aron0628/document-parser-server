"""LLM 프로바이더 라우팅 단위 테스트"""

import io
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# psycopg_pool / pgvector가 테스트 환경에 없을 수 있으므로 stub 등록
def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "psycopg_pool" not in sys.modules:
    _psycopg_pool = _stub_module("psycopg_pool")
    _psycopg_pool.AsyncConnectionPool = MagicMock

if "pgvector" not in sys.modules:
    _stub_module("pgvector")
    _stub_module("pgvector.psycopg")
    sys.modules["pgvector.psycopg"].register_vector_async = AsyncMock()


# ---------------------------------------------------------------------------
# parse_model_string 단위 테스트
# ---------------------------------------------------------------------------


def test_parse_model_string_openai():
    """openai/gpt-4o → ('openai', 'gpt-4o') 파싱"""
    from app.pipeline.external.llm_provider import parse_model_string

    assert parse_model_string("openai/gpt-4o") == ("openai", "gpt-4o")


def test_parse_model_string_gemini():
    """gemini/gemini-2.5-flash → ('gemini', 'gemini-2.5-flash') 파싱"""
    from app.pipeline.external.llm_provider import parse_model_string

    assert parse_model_string("gemini/gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")


def test_parse_model_string_grok():
    """grok/grok-3 → ('grok', 'grok-3') 파싱"""
    from app.pipeline.external.llm_provider import parse_model_string

    assert parse_model_string("grok/grok-3") == ("grok", "grok-3")


def test_parse_model_string_slash_in_model_name():
    """슬래시가 여러 개인 경우 첫 번째 슬래시 기준으로만 분리"""
    from app.pipeline.external.llm_provider import parse_model_string

    provider, model = parse_model_string("openai/gpt-4/turbo")
    assert provider == "openai"
    assert model == "gpt-4/turbo"


def test_parse_model_string_no_slash_raises():
    """슬래시 없는 문자열 → ValueError 발생"""
    from app.pipeline.external.llm_provider import parse_model_string

    with pytest.raises(ValueError, match="프로바이더 접두사"):
        parse_model_string("gpt-4o")


def test_parse_model_string_empty_raises():
    """빈 문자열 → ValueError 발생"""
    from app.pipeline.external.llm_provider import parse_model_string

    with pytest.raises(ValueError):
        parse_model_string("")


# ---------------------------------------------------------------------------
# validate_model 단위 테스트
# ---------------------------------------------------------------------------


def test_validate_model_allowed_vision():
    """허용된 vision 모델이면 예외 없음"""
    from app.pipeline.external.llm_provider import ALLOWED_VISION_MODELS, validate_model

    validate_model("openai/gpt-4o", ALLOWED_VISION_MODELS)  # 예외 없어야 함


def test_validate_model_allowed_text():
    """허용된 text 모델이면 예외 없음"""
    from app.pipeline.external.llm_provider import ALLOWED_TEXT_MODELS, validate_model

    validate_model("gemini/gemini-2.5-flash", ALLOWED_TEXT_MODELS)  # 예외 없어야 함


def test_validate_model_not_allowed_raises():
    """허용 목록에 없는 모델 → ValueError 발생"""
    from app.pipeline.external.llm_provider import ALLOWED_VISION_MODELS, validate_model

    with pytest.raises(ValueError, match="허용되지 않는 모델"):
        validate_model("unknown/model", ALLOWED_VISION_MODELS)


def test_validate_model_wrong_provider_raises():
    """존재하지 않는 프로바이더 모델 → ValueError 발생"""
    from app.pipeline.external.llm_provider import ALLOWED_TEXT_MODELS, validate_model

    with pytest.raises(ValueError):
        validate_model("anthropic/claude-3-5-sonnet", ALLOWED_TEXT_MODELS)


# ---------------------------------------------------------------------------
# resolve_api_key 단위 테스트
# ---------------------------------------------------------------------------


def test_resolve_api_key_openai():
    """openai 프로바이더 → openai_api_key 반환"""
    from app.pipeline.external.llm_provider import resolve_api_key

    keys = {"openai_api_key": "sk-xxx", "google_api_key": "", "xai_api_key": ""}
    assert resolve_api_key("openai", keys) == "sk-xxx"


def test_resolve_api_key_gemini():
    """gemini 프로바이더 → google_api_key 반환"""
    from app.pipeline.external.llm_provider import resolve_api_key

    keys = {"openai_api_key": "", "google_api_key": "AI-xxx", "xai_api_key": ""}
    assert resolve_api_key("gemini", keys) == "AI-xxx"


def test_resolve_api_key_grok():
    """grok 프로바이더 → xai_api_key 반환"""
    from app.pipeline.external.llm_provider import resolve_api_key

    keys = {"openai_api_key": "", "google_api_key": "", "xai_api_key": "xai-yyy"}
    assert resolve_api_key("grok", keys) == "xai-yyy"


def test_resolve_api_key_empty_raises():
    """해당 프로바이더 키가 빈 문자열 → ValueError 발생"""
    from app.pipeline.external.llm_provider import resolve_api_key

    keys = {"openai_api_key": "", "google_api_key": "", "xai_api_key": ""}
    with pytest.raises(ValueError, match="API 키가 없습니다"):
        resolve_api_key("openai", keys)


def test_resolve_api_key_missing_key_raises():
    """keys dict에 해당 필드가 아예 없을 때도 ValueError 발생"""
    from app.pipeline.external.llm_provider import resolve_api_key

    with pytest.raises(ValueError):
        resolve_api_key("gemini", {})


def test_resolve_api_key_unknown_provider_raises():
    """미등록 프로바이더 → ValueError 발생 (get_provider_config 에서)"""
    from app.pipeline.external.llm_provider import resolve_api_key

    with pytest.raises(ValueError, match="지원하지 않는 프로바이더"):
        resolve_api_key("anthropic", {"anthropic_api_key": "sk-ant-xxx"})


# ---------------------------------------------------------------------------
# get_provider_config 단위 테스트
# ---------------------------------------------------------------------------


def test_get_provider_config_openai():
    """openai 프로바이더 설정 조회"""
    from app.pipeline.external.llm_provider import get_provider_config

    config = get_provider_config("openai")
    assert config.base_url == "https://api.openai.com/v1"
    assert config.supports_detail is True
    assert config.key_field == "openai_api_key"


def test_get_provider_config_gemini():
    """gemini 프로바이더 설정 조회"""
    from app.pipeline.external.llm_provider import get_provider_config

    config = get_provider_config("gemini")
    assert "generativelanguage" in config.base_url
    assert config.supports_detail is False
    assert config.key_field == "google_api_key"


def test_get_provider_config_grok():
    """grok 프로바이더 설정 조회"""
    from app.pipeline.external.llm_provider import get_provider_config

    config = get_provider_config("grok")
    assert "x.ai" in config.base_url
    assert config.supports_detail is False
    assert config.key_field == "xai_api_key"


def test_get_provider_config_invalid_raises():
    """미등록 프로바이더 → ValueError 발생"""
    from app.pipeline.external.llm_provider import get_provider_config

    with pytest.raises(ValueError, match="지원하지 않는 프로바이더"):
        get_provider_config("unknown")


# ---------------------------------------------------------------------------
# build_vision_image_block 단위 테스트
# ---------------------------------------------------------------------------


def test_build_vision_image_block_openai_has_detail():
    """openai 프로바이더 → detail 파라미터 포함"""
    from app.pipeline.external.llm_provider import build_vision_image_block

    block = build_vision_image_block("data:image/png;base64,abc", "openai")

    assert block["type"] == "image_url"
    assert block["image_url"]["url"] == "data:image/png;base64,abc"
    assert block["image_url"]["detail"] == "low"


def test_build_vision_image_block_gemini_no_detail():
    """gemini 프로바이더 → detail 파라미터 미포함"""
    from app.pipeline.external.llm_provider import build_vision_image_block

    block = build_vision_image_block("data:image/png;base64,abc", "gemini")

    assert block["type"] == "image_url"
    assert block["image_url"]["url"] == "data:image/png;base64,abc"
    assert "detail" not in block["image_url"]


def test_build_vision_image_block_grok_no_detail():
    """grok 프로바이더 → detail 파라미터 미포함"""
    from app.pipeline.external.llm_provider import build_vision_image_block

    block = build_vision_image_block("data:image/png;base64,abc", "grok")

    assert "detail" not in block["image_url"]


def test_build_vision_image_block_url_preserved():
    """b64_url이 그대로 image_url.url에 들어가는지 확인"""
    from app.pipeline.external.llm_provider import build_vision_image_block

    url = "data:image/jpeg;base64,/9j/4AAQ"
    block = build_vision_image_block(url, "gemini")
    assert block["image_url"]["url"] == url


# ---------------------------------------------------------------------------
# _call_openai base_url 전달 테스트 (httpx mock)
# ---------------------------------------------------------------------------


async def test_call_openai_custom_base_url():
    """base_url 파라미터가 올바르게 요청 URL에 반영되는지 확인"""
    from app.pipeline.external.openai_client import _call_openai

    gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    captured_urls = []

    # httpx.AsyncClient.post를 mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "응답"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    async def fake_post(url, **kwargs):
        captured_urls.append(url)
        return mock_response

    mock_client_instance.post = fake_post

    with patch("app.pipeline.external.openai_client.httpx.AsyncClient", return_value=mock_client_instance):
        result = await _call_openai(
            api_key="test-key",
            messages=[{"role": "user", "content": "안녕"}],
            model="gemini-2.5-flash",
            base_url=gemini_base_url,
        )

    assert len(captured_urls) == 1
    assert captured_urls[0] == f"{gemini_base_url}/chat/completions"
    assert result["choices"][0]["message"]["content"] == "응답"


async def test_call_openai_default_base_url():
    """base_url 기본값이 OpenAI 엔드포인트인지 확인"""
    from app.pipeline.external.openai_client import _call_openai

    captured_urls = []

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "응답"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    async def fake_post(url, **kwargs):
        captured_urls.append(url)
        return mock_response

    mock_client_instance.post = fake_post

    with patch("app.pipeline.external.openai_client.httpx.AsyncClient", return_value=mock_client_instance):
        await _call_openai(
            api_key="test-key",
            messages=[{"role": "user", "content": "안녕"}],
        )

    assert len(captured_urls) == 1
    assert "api.openai.com" in captured_urls[0]


async def test_call_openai_grok_base_url():
    """grok base_url로 호출 시 URL이 x.ai를 포함하는지 확인"""
    from app.pipeline.external.openai_client import _call_openai

    grok_base_url = "https://api.x.ai/v1"
    captured_urls = []

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "응답"}}]}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    async def fake_post(url, **kwargs):
        captured_urls.append(url)
        return mock_response

    mock_client_instance.post = fake_post

    with patch("app.pipeline.external.openai_client.httpx.AsyncClient", return_value=mock_client_instance):
        await _call_openai(
            api_key="xai-key",
            messages=[{"role": "user", "content": "안녕"}],
            model="grok-3",
            base_url=grok_base_url,
        )

    assert captured_urls[0] == f"{grok_base_url}/chat/completions"


async def test_call_openai_bearer_token_in_header():
    """Authorization 헤더에 Bearer 토큰이 올바르게 전달되는지 확인"""
    from app.pipeline.external.openai_client import _call_openai

    captured_headers = []

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    async def fake_post(url, headers=None, **kwargs):
        captured_headers.append(headers)
        return mock_response

    mock_client_instance.post = fake_post

    with patch("app.pipeline.external.openai_client.httpx.AsyncClient", return_value=mock_client_instance):
        await _call_openai(api_key="my-secret-key", messages=[])

    assert captured_headers[0]["Authorization"] == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# routes.py fail-fast 검증 테스트
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """FastAPI TestClient fixture"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_parse_invalid_vision_model_returns_400(client):
    """미지원 vision_model이 설정되어 있으면 /parse가 400 반환"""
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")

    with patch("app.api.routes.get_app_setting") as mock_setting, \
         patch("app.db.get_app_setting_int", return_value=100), \
         patch("app.api.routes.get_app_setting_bool", return_value=True), \
         patch("app.api.routes.settings") as mock_settings:
        # vision_model은 허용 목록에 없는 값, raptor_model은 유효한 값
        def setting_side_effect(key, default=""):
            if key == "vision_model":
                return "unknown/invalid-model"
            if key == "raptor_summarization_model":
                return "openai/gpt-4.1-mini"
            return default

        mock_setting.side_effect = setting_side_effect
        mock_settings.upstage_api_key = ""
        mock_settings.openai_api_key = ""
        mock_settings.google_api_key = ""
        mock_settings.xai_api_key = ""
        mock_settings.vision_model = "unknown/invalid-model"
        mock_settings.raptor_summarization_model = "openai/gpt-4.1-mini"
        mock_settings.default_embedding_model = "solar-embedding-1-large-passage"
        mock_settings.chunk_size = 1000
        mock_settings.chunk_overlap = 200

        resp = client.post(
            "/parse",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            headers={
                "X-UPSTAGE-API-KEY": "upstage-test",
                "X-OPENAI-API-KEY": "openai-test",
            },
        )

    assert resp.status_code == 400
    assert "허용되지 않는 모델" in resp.json()["detail"]


def test_parse_invalid_raptor_model_returns_400(client):
    """미지원 raptor_summarization_model이 설정되어 있으면 /parse가 400 반환"""
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")

    with patch("app.api.routes.get_app_setting") as mock_setting, \
         patch("app.db.get_app_setting_int", return_value=100), \
         patch("app.api.routes.get_app_setting_bool", return_value=True), \
         patch("app.api.routes.settings") as mock_settings:
        def setting_side_effect(key, default=""):
            if key == "vision_model":
                return "openai/gpt-4o"
            if key == "raptor_summarization_model":
                return "unknown/bad-model"
            return default

        mock_setting.side_effect = setting_side_effect
        mock_settings.upstage_api_key = ""
        mock_settings.openai_api_key = ""
        mock_settings.google_api_key = ""
        mock_settings.xai_api_key = ""
        mock_settings.vision_model = "openai/gpt-4o"
        mock_settings.raptor_summarization_model = "unknown/bad-model"
        mock_settings.default_embedding_model = "solar-embedding-1-large-passage"
        mock_settings.chunk_size = 1000
        mock_settings.chunk_overlap = 200

        resp = client.post(
            "/parse",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            headers={
                "X-UPSTAGE-API-KEY": "upstage-test",
                "X-OPENAI-API-KEY": "openai-test",
            },
        )

    assert resp.status_code == 400
    assert "허용되지 않는 모델" in resp.json()["detail"]


def test_parse_valid_model_passes_validation(client):
    """지원 모델이 설정되어 있으면 모델 검증 통과 (파이프라인 실행 전까지)"""
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")

    with patch("app.api.routes.get_app_setting") as mock_setting, \
         patch("app.db.get_app_setting_int", return_value=100), \
         patch("app.api.routes.get_app_setting_bool", return_value=True), \
         patch("app.api.routes.settings") as mock_settings, \
         patch("app.api.routes.job_manager") as mock_jm, \
         patch("app.api.routes.file_manager") as mock_fm, \
         patch("app.api.routes._run_pipeline", new_callable=AsyncMock):
        def setting_side_effect(key, default=""):
            if key == "vision_model":
                return "gemini/gemini-2.5-flash"
            if key == "raptor_summarization_model":
                return "openai/gpt-4.1-mini"
            return default

        mock_setting.side_effect = setting_side_effect
        mock_settings.upstage_api_key = ""
        mock_settings.openai_api_key = ""
        mock_settings.google_api_key = ""
        mock_settings.xai_api_key = ""
        mock_settings.vision_model = "gemini/gemini-2.5-flash"
        mock_settings.raptor_summarization_model = "openai/gpt-4.1-mini"
        mock_settings.default_embedding_model = "solar-embedding-1-large-passage"
        mock_settings.chunk_size = 1000
        mock_settings.chunk_overlap = 200

        mock_jm.create_job.return_value = {"job_id": "test-job-valid", "status": "pending"}
        mock_fm.save_upload = AsyncMock(return_value="/tmp/test.pdf")

        resp = client.post(
            "/parse",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            headers={
                "X-UPSTAGE-API-KEY": "upstage-test",
                "X-OPENAI-API-KEY": "openai-test",
            },
        )

    # 400이 아니어야 함 (모델 검증 통과)
    assert resp.status_code != 400 or "허용되지 않는 모델" not in resp.json().get("detail", "")


def test_parse_missing_api_keys_returns_400(client):
    """API 키 미제공 → 400 반환"""
    fake_pdf = io.BytesIO(b"%PDF-1.4 content")

    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.upstage_api_key = ""
        mock_settings.openai_api_key = ""
        mock_settings.google_api_key = ""
        mock_settings.xai_api_key = ""

        resp = client.post(
            "/parse",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            # 헤더 없음
        )

    assert resp.status_code == 400
