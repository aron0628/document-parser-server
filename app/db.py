"""PostgreSQL 커넥션 풀 및 DB 초기화 모듈"""

import logging
from typing import Optional
from urllib.parse import quote_plus

import psycopg_pool
from pgvector.psycopg import register_vector_async

from app.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[psycopg_pool.AsyncConnectionPool] = None

_CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_embeddings (
    id BIGSERIAL PRIMARY KEY,
    job_id VARCHAR(64) NOT NULL,
    element_index INTEGER NOT NULL,
    page INTEGER,
    element_type VARCHAR(32),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(4096) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEX_JOB_ID_SQL = """
CREATE INDEX IF NOT EXISTS idx_doc_emb_job_id ON document_embeddings (job_id);
"""

_CREATE_INDEX_JOB_TYPE_SQL = """
CREATE INDEX IF NOT EXISTS idx_doc_emb_job_type ON document_embeddings (job_id, element_type);
"""


async def init_db() -> None:
    """커넥션 풀 생성, vector 타입 등록, 테이블/인덱스 생성"""
    global _pool

    logger.info("DB 초기화 시작")

    database_url = (
        f"postgresql://{quote_plus(settings.db_user)}:{quote_plus(settings.db_password)}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )

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

    async with _pool.connection() as conn:
        await register_vector_async(conn)
        logger.info("vector 타입 등록 완료")

        await conn.execute(_CREATE_EXTENSION_SQL)
        await conn.execute(_CREATE_TABLE_SQL)
        await conn.execute(_CREATE_INDEX_JOB_ID_SQL)
        await conn.execute(_CREATE_INDEX_JOB_TYPE_SQL)
        await conn.commit()
        logger.info("테이블 및 인덱스 생성 완료")

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
