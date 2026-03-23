"""Checkpointer 초기화/정리 테스트"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# psycopg / langgraph.checkpoint.postgres가 테스트 환경에 없을 수 있으므로 stub 등록
def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "psycopg" not in sys.modules:
    _psycopg = _stub_module("psycopg")
    _psycopg.AsyncConnection = MagicMock()

if "langgraph.checkpoint.postgres" not in sys.modules:
    _stub_module("langgraph.checkpoint.postgres")
if "langgraph.checkpoint.postgres.aio" not in sys.modules:
    _aio = _stub_module("langgraph.checkpoint.postgres.aio")
    _aio.AsyncPostgresSaver = MagicMock()


# ---------------------------------------------------------------------------
# 헬퍼: 모듈 전역 상태 초기화
# ---------------------------------------------------------------------------


def _reset_checkpointer_state():
    """테스트 격리를 위해 checkpointer 모듈 전역 변수 초기화"""
    import app.pipeline.checkpointer as cp_module

    cp_module._checkpointer = None
    cp_module._connection = None


# ---------------------------------------------------------------------------
# TestInitCheckpointer
# ---------------------------------------------------------------------------


class TestInitCheckpointer:
    """checkpointer 초기화 테스트"""

    def setup_method(self):
        _reset_checkpointer_state()

    def teardown_method(self):
        _reset_checkpointer_state()

    async def test_init_success(self):
        """DB 연결 성공 시 checkpointer 초기화"""
        mock_connection = AsyncMock()
        mock_saver = MagicMock()

        mock_async_conn_cls = MagicMock()
        mock_async_conn_cls.connect = AsyncMock(return_value=mock_connection)

        mock_saver_cls = MagicMock(return_value=mock_saver)

        with patch("app.pipeline.checkpointer.settings") as mock_settings, \
             patch("psycopg.AsyncConnection", mock_async_conn_cls), \
             patch("langgraph.checkpoint.postgres.aio.AsyncPostgresSaver", mock_saver_cls):
            mock_settings.enable_checkpointer = True
            mock_settings.db_user = "user"
            mock_settings.db_password = "pass"
            mock_settings.db_host = "localhost"
            mock_settings.db_port = 5432
            mock_settings.db_name = "testdb"

            from app.pipeline.checkpointer import init_checkpointer, get_checkpointer

            await init_checkpointer()

        assert get_checkpointer() is not None

    async def test_init_disabled(self):
        """enable_checkpointer=False 시 스킵"""
        with patch("app.pipeline.checkpointer.settings") as mock_settings:
            mock_settings.enable_checkpointer = False

            from app.pipeline.checkpointer import init_checkpointer, get_checkpointer

            await init_checkpointer()

        assert get_checkpointer() is None

    async def test_init_db_failure(self):
        """DB 연결 실패 시 graceful degradation — 에러 없이 None 유지"""
        mock_async_conn_cls = MagicMock()
        mock_async_conn_cls.connect = AsyncMock(side_effect=Exception("연결 실패"))

        with patch("app.pipeline.checkpointer.settings") as mock_settings, \
             patch("psycopg.AsyncConnection", mock_async_conn_cls):
            mock_settings.enable_checkpointer = True
            mock_settings.db_user = "user"
            mock_settings.db_password = "pass"
            mock_settings.db_host = "localhost"
            mock_settings.db_port = 5432
            mock_settings.db_name = "testdb"

            from app.pipeline.checkpointer import init_checkpointer, get_checkpointer

            # 예외가 전파되면 안 됨
            await init_checkpointer()

        assert get_checkpointer() is None


# ---------------------------------------------------------------------------
# TestCompileGraph
# ---------------------------------------------------------------------------


class TestCompileGraph:
    """compile_graph 테스트"""

    def test_compile_without_checkpointer(self):
        """checkpointer=None으로 컴파일 — 기존 동작과 동일"""
        from app.pipeline.graph import compile_graph

        graph = compile_graph(checkpointer=None)
        assert graph is not None

    def test_compile_with_checkpointer(self):
        """checkpointer 전달 시 정상 컴파일"""
        from langgraph.checkpoint.memory import InMemorySaver
        from app.pipeline.graph import compile_graph

        checkpointer = InMemorySaver()
        graph = compile_graph(checkpointer=checkpointer)
        assert graph is not None


# ---------------------------------------------------------------------------
# TestCleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """checkpoint 정리 테스트"""

    def setup_method(self):
        _reset_checkpointer_state()

    def teardown_method(self):
        _reset_checkpointer_state()

    async def test_cleanup_job_when_no_connection(self):
        """connection=None이면 에러 없이 스킵"""
        from app.pipeline.checkpointer import cleanup_job_checkpoints

        # _connection이 None인 상태 (setup_method에서 초기화됨)
        await cleanup_job_checkpoints("test-job-id")  # 예외 없이 정상 반환

