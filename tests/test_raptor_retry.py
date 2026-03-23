"""RAPTOR retry 관련 테스트 (_finalize_job, /retry-raptor, StatusResponse)"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# psycopg_pool / pgvector가 테스트 환경에 없을 수 있으므로 stub 등록
def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "psycopg_pool" not in sys.modules:
    _psycopg_pool = _stub_module("psycopg_pool")
    _psycopg_pool.AsyncConnectionPool = MagicMock

if "pgvector" not in sys.modules:
    from unittest.mock import AsyncMock
    _stub_module("pgvector")
    _stub_module("pgvector.psycopg")
    sys.modules["pgvector.psycopg"].register_vector_async = AsyncMock()


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# TestFinalizeJobRaptorStatus
# ---------------------------------------------------------------------------


class TestFinalizeJobRaptorStatus:
    """_finalize_job의 raptor_status 판단 로직 테스트"""

    def _call_finalize(self, result_state: dict, job_data: dict):
        """_finalize_job을 호출하고 update_job에 전달된 kwargs 반환"""
        from app.api.routes import _finalize_job

        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.api.routes.create_result_zip", return_value="result.zip", create=True), \
             patch("app.utils.zip_utils.create_result_zip", return_value="result.zip"):
            mock_jm.get_job.return_value = job_data
            _finalize_job("test-job-id", result_state)
            return mock_jm.update_job.call_args

    def test_raptor_enabled_success(self):
        """enable_raptor=True, raptor_level_counts 비어있지 않음 → success"""
        from app.api.routes import _finalize_job

        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.utils.zip_utils.create_result_zip", return_value="result.zip"):
            mock_jm.get_job.return_value = {"filename": "test.pdf"}
            result_state = {
                "enable_raptor": True,
                "raptor_level_counts": {"0": 5, "1": 2},
            }
            _finalize_job("job-1", result_state)

        _, kwargs = mock_jm.update_job.call_args
        assert kwargs["raptor_status"] == "success"

    def test_raptor_enabled_failed(self):
        """enable_raptor=True, raptor_level_counts={} → failed"""
        from app.api.routes import _finalize_job

        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.utils.zip_utils.create_result_zip", return_value="result.zip"):
            mock_jm.get_job.return_value = {"filename": "test.pdf"}
            result_state = {
                "enable_raptor": True,
                "raptor_level_counts": {},
            }
            _finalize_job("job-2", result_state)

        _, kwargs = mock_jm.update_job.call_args
        assert kwargs["raptor_status"] == "failed"

    def test_raptor_disabled_skipped(self):
        """enable_raptor=False → skipped"""
        from app.api.routes import _finalize_job

        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.utils.zip_utils.create_result_zip", return_value="result.zip"):
            mock_jm.get_job.return_value = {"filename": "test.pdf"}
            result_state = {
                "enable_raptor": False,
                "raptor_level_counts": {"0": 3},
            }
            _finalize_job("job-3", result_state)

        _, kwargs = mock_jm.update_job.call_args
        assert kwargs["raptor_status"] == "skipped"

    def test_raptor_not_set_skipped(self):
        """enable_raptor 키 없음 (기본값 False) → skipped"""
        from app.api.routes import _finalize_job

        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.utils.zip_utils.create_result_zip", return_value="result.zip"):
            mock_jm.get_job.return_value = {"filename": "test.pdf"}
            result_state = {}
            _finalize_job("job-4", result_state)

        _, kwargs = mock_jm.update_job.call_args
        assert kwargs["raptor_status"] == "skipped"


# ---------------------------------------------------------------------------
# TestRetryRaptorEndpoint
# ---------------------------------------------------------------------------


class TestRetryRaptorEndpoint:
    """POST /retry-raptor/{job_id} 테스트"""

    def test_retry_raptor_job_not_found(self, client):
        """존재하지 않는 job_id → 404"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = None
            resp = client.post(
                "/retry-raptor/nonexistent",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 404

    def test_retry_raptor_not_completed(self, client):
        """completed가 아닌 job (processing) → 400"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "job-proc",
                "status": "processing",
                "filename": "test.pdf",
            }
            resp = client.post(
                "/retry-raptor/job-proc",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 400

    def test_retry_raptor_not_completed_failed(self, client):
        """completed가 아닌 job (failed) → 400"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "job-fail",
                "status": "failed",
                "filename": "test.pdf",
            }
            resp = client.post(
                "/retry-raptor/job-fail",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 400

    def test_retry_raptor_already_processing(self, client):
        """raptor_status=processing → 409"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "job-rp",
                "status": "completed",
                "filename": "test.pdf",
                "raptor_status": "processing",
            }
            resp = client.post(
                "/retry-raptor/job-rp",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 409

    def test_retry_raptor_already_success(self, client):
        """raptor_status=success → 400"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "job-rs",
                "status": "completed",
                "filename": "test.pdf",
                "raptor_status": "success",
            }
            resp = client.post(
                "/retry-raptor/job-rs",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 400

    def test_retry_raptor_no_api_keys(self, client):
        """API 키 미제공 + env 미설정 → 400"""
        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.api.routes.settings") as mock_settings:
            mock_jm.get_job.return_value = {
                "job_id": "job-nokey",
                "status": "completed",
                "filename": "test.pdf",
                "raptor_status": "failed",
            }
            mock_settings.upstage_api_key = ""
            mock_settings.openai_api_key = ""
            resp = client.post("/retry-raptor/job-nokey")  # 헤더 없음
        assert resp.status_code == 400

    def test_retry_raptor_success(self, client):
        """정상 요청 → 200, raptor_status=processing"""
        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.api.routes._run_raptor_retry"):
            mock_jm.get_job.return_value = {
                "job_id": "job-ok",
                "status": "completed",
                "filename": "test.pdf",
                "raptor_status": "failed",
            }
            resp = client.post(
                "/retry-raptor/job-ok",
                headers={"X-UPSTAGE-API-KEY": "upstage-key", "X-OPENAI-API-KEY": "openai-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-ok"
        assert data["raptor_status"] == "processing"

    def test_retry_raptor_success_skipped_status(self, client):
        """raptor_status=skipped인 job도 재시도 가능 → 200"""
        with patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.api.routes._run_raptor_retry"):
            mock_jm.get_job.return_value = {
                "job_id": "job-skip",
                "status": "completed",
                "filename": "test.pdf",
                "raptor_status": "skipped",
            }
            resp = client.post(
                "/retry-raptor/job-skip",
                headers={"X-UPSTAGE-API-KEY": "upstage-key", "X-OPENAI-API-KEY": "openai-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["raptor_status"] == "processing"


# ---------------------------------------------------------------------------
# TestStatusResponseRaptorFields
# ---------------------------------------------------------------------------


class TestStatusResponseRaptorFields:
    """GET /status/{job_id} 응답에 raptor 필드 포함 여부 테스트"""

    def test_status_includes_raptor_status(self, client):
        """raptor_status 필드가 응답에 포함됨"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "job-s1",
                "status": "completed",
                "filename": "test.pdf",
                "created_at": 1700000000.0,
                "completed_at": 1700001000.0,
                "raptor_status": "success",
                "raptor_error": None,
            }
            resp = client.get("/status/job-s1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["raptor_status"] == "success"
        assert data["raptor_error"] is None

    def test_status_includes_raptor_error(self, client):
        """raptor_error 필드가 응답에 포함됨"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "job-s2",
                "status": "completed",
                "filename": "test.pdf",
                "created_at": 1700000000.0,
                "raptor_status": "failed",
                "raptor_error": "RAPTOR 처리 중 오류 발생",
            }
            resp = client.get("/status/job-s2")

        assert resp.status_code == 200
        data = resp.json()
        assert data["raptor_status"] == "failed"
        assert data["raptor_error"] == "RAPTOR 처리 중 오류 발생"

    def test_status_raptor_fields_none_when_absent(self, client):
        """raptor 필드 없는 job → None 반환"""
        with patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "job-s3",
                "status": "processing",
                "filename": "test.pdf",
                "created_at": 1700000000.0,
            }
            resp = client.get("/status/job-s3")

        assert resp.status_code == 200
        data = resp.json()
        assert data["raptor_status"] is None
        assert data["raptor_error"] is None
