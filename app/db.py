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
