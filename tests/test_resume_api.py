"""Resume API endpoint 테스트"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

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
# TestResumeEndpoint
# ---------------------------------------------------------------------------


class TestResumeEndpoint:
    """POST /resume/{job_id} 테스트"""

    def test_resume_no_checkpointer(self, client):
        """checkpointer 미활성화 → 503"""
        # routes.py는 함수 내부에서 `from app.pipeline.checkpointer import get_checkpointer` 호출
        # 따라서 모듈 경로를 직접 패치
        with patch("app.pipeline.checkpointer.get_checkpointer", return_value=None):
            resp = client.post(
                "/resume/any-job-id",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 503

    def test_resume_job_not_found(self, client):
        """존재하지 않는 job_id → 404"""
        with patch("app.pipeline.checkpointer.get_checkpointer", return_value=MagicMock()), \
             patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = None
            resp = client.post(
                "/resume/nonexistent",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 404

    def test_resume_not_failed_job(self, client):
        """failed가 아닌 job (completed) → 400"""
        with patch("app.pipeline.checkpointer.get_checkpointer", return_value=MagicMock()), \
             patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "123",
                "status": "completed",
                "filename": "test.pdf",
            }
            resp = client.post(
                "/resume/123",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 400

    def test_resume_not_failed_job_processing(self, client):
        """failed가 아닌 job (processing) → 400"""
        with patch("app.pipeline.checkpointer.get_checkpointer", return_value=MagicMock()), \
             patch("app.api.routes.job_manager") as mock_jm:
            mock_jm.get_job.return_value = {
                "job_id": "456",
                "status": "processing",
                "filename": "test.pdf",
            }
            resp = client.post(
                "/resume/456",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 400

    def test_resume_no_api_keys(self, client):
        """API 키 없음 → 400"""
        with patch("app.pipeline.checkpointer.get_checkpointer", return_value=MagicMock()), \
             patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.api.routes.settings") as mock_settings:
            mock_jm.get_job.return_value = {
                "job_id": "789",
                "status": "failed",
                "filename": "test.pdf",
            }
            mock_settings.upstage_api_key = ""
            mock_settings.openai_api_key = ""
            resp = client.post("/resume/789")  # 헤더 없음
        assert resp.status_code == 400

    def test_resume_no_checkpoint_saved(self, client):
        """job은 failed지만 저장된 checkpoint 없음 → 404"""
        mock_runner = MagicMock()
        # aget_state가 빈 state(values 없음) 반환
        mock_saved_state = MagicMock()
        mock_saved_state.values = {}
        mock_runner._graph.aget_state = AsyncMock(return_value=mock_saved_state)

        with patch("app.pipeline.checkpointer.get_checkpointer", return_value=MagicMock()), \
             patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.api.routes.settings") as mock_settings, \
             patch("app.pipeline.runner.get_runner", return_value=mock_runner):
            mock_jm.get_job.return_value = {
                "job_id": "abc",
                "status": "failed",
                "filename": "test.pdf",
            }
            mock_settings.upstage_api_key = "upstage-key"
            mock_settings.openai_api_key = "openai-key"
            resp = client.post(
                "/resume/abc",
                headers={"X-UPSTAGE-API-KEY": "test", "X-OPENAI-API-KEY": "test"},
            )
        assert resp.status_code == 404

    def test_resume_success(self, client):
        """정상 재개 요청 → 200, status=processing"""
        mock_runner = MagicMock()
        mock_saved_state = MagicMock()
        mock_saved_state.values = {"pdf_chunks": ["chunk1", "chunk2"]}
        mock_runner._graph.aget_state = AsyncMock(return_value=mock_saved_state)

        with patch("app.pipeline.checkpointer.get_checkpointer", return_value=MagicMock()), \
             patch("app.api.routes.job_manager") as mock_jm, \
             patch("app.api.routes.settings") as mock_settings, \
             patch("app.pipeline.runner.get_runner", return_value=mock_runner), \
             patch("app.api.routes._resume_pipeline", new_callable=AsyncMock):
            mock_jm.get_job.return_value = {
                "job_id": "ok-job",
                "status": "failed",
                "filename": "test.pdf",
            }
            mock_settings.upstage_api_key = ""
            mock_settings.openai_api_key = ""
            resp = client.post(
                "/resume/ok-job",
                headers={"X-UPSTAGE-API-KEY": "upstage-key", "X-OPENAI-API-KEY": "openai-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "ok-job"
        assert data["status"] == "processing"
