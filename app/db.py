"""PostgreSQL 커넥션 풀 및 DB 초기화 모듈"""

import logging
from typing import Optional
from urllib.parse import quote_plus

import psycopg_pool
from pgvector.psycopg import register_vector_async

from app.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[psycopg_pool.AsyncConnectionPool] = None


async def init_db() -> None:
    """커넥션 풀 생성 및 vector 타입 등록

    DB 연결에 실패해도 서버는 정상 기동된다.
    임베딩 기능만 비활성화 상태로 동작한다.
    """
    global _pool

    logger.info("DB 초기화 시작")

    database_url = (
        f"postgresql://{quote_plus(settings.db_user)}:{quote_plus(settings.db_password)}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )

    try:
        _pool = psycopg_pool.AsyncConnectionPool(
            database_url,
            open=False,
            min_size=4,
            max_size=10,
            max_idle=300.0,
            check=psycopg_pool.AsyncConnectionPool.check_connection,
        )
        await _pool.open()
        logger.info("커넥션 풀 생성 완료")
    except Exception as e:
        logger.warning(f"DB 커넥션 풀 생성 실패 (임베딩 비활성화): {e}")
        _pool = None
        return

    try:
        async with _pool.connection() as conn:
            await register_vector_async(conn)
            logger.info("vector 타입 등록 완료")
    except Exception as e:
        logger.warning(f"vector 타입 등록 실패 (임베딩 비활성화): {e}")
        await _pool.close()
        _pool = None
        return

    # RAPTOR summaries 테이블 존재 확인 (DDL 권한 불필요)
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM raptor_summaries LIMIT 0"
                )
            logger.info("raptor_summaries 테이블 확인 완료")
    except Exception as e:
        logger.warning(f"raptor_summaries 테이블 미존재 (RAPTOR 비활성화): {e}")

    logger.info("DB 초기화 완료")


def get_pool() -> psycopg_pool.AsyncConnectionPool:
    """커넥션 풀 반환"""
    if _pool is None:
        raise RuntimeError("DB 풀이 초기화되지 않았습니다. init_db()를 먼저 호출하세요.")
    return _pool


async def close_pool() -> None:
    """커넥션 풀 종료"""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("커넥션 풀 종료 완료")


# ---------------------------------------------------------------------------
# app_settings 테이블 공유 설정 조회
# ---------------------------------------------------------------------------
_app_settings_cache: dict[str, str] = {}


async def load_app_settings() -> None:
    """app_settings 테이블에서 설정을 로드하여 캐시."""
    global _app_settings_cache

    if _pool is None:
        logger.warning("load_app_settings: DB 풀 없음, 기본값 사용")
        return

    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT key, value FROM app_settings")
                rows = await cur.fetchall()
                _app_settings_cache = {row[0]: row[1] for row in rows}
        logger.info(f"app_settings 로드 완료 ({len(_app_settings_cache)}건)")
    except Exception as e:
        logger.warning(f"app_settings 로드 실패 (기본값 사용): {e}")
        _app_settings_cache = {}


def get_app_setting(key: str, default: str = "") -> str:
    """캐시된 app_settings 값 반환."""
    return _app_settings_cache.get(key, default)


def get_app_setting_int(key: str, default: int = 0) -> int:
    """캐시된 app_settings 값을 int로 반환."""
    val = _app_settings_cache.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default
