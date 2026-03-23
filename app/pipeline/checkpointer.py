"""LangGraph Checkpointer 관리 모듈

AsyncPostgresSaver 싱글턴 초기화/반환/종료.
DB 실패 시 None으로 fallback (graceful degradation).
"""

import logging
from urllib.parse import quote_plus

from app.config import settings

logger = logging.getLogger(__name__)

_checkpointer = None
_connection = None  # psycopg AsyncConnection


async def init_checkpointer() -> None:
    """AsyncPostgresSaver 초기화. DB 실패 시 None 유지 (warning 로그)."""
    global _checkpointer, _connection

    if not settings.enable_checkpointer:
        logger.info("Checkpointer 비활성화 (enable_checkpointer=False)")
        return

    try:
        import psycopg
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        database_url = (
            f"postgresql://{quote_plus(settings.db_user)}:{quote_plus(settings.db_password)}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        )

        # AsyncPostgresSaver는 단일 AsyncConnection을 사용
        # TCP keepalive: 장시간 유휴 시 커넥션 끊김 방지 (RAPTOR 등 수 분 소요 노드)
        _connection = await psycopg.AsyncConnection.connect(
            database_url,
            autocommit=True,
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=15,
            keepalives_count=4,
        )
        _checkpointer = AsyncPostgresSaver(conn=_connection)

        logger.info("Checkpointer 초기화 완료 (AsyncPostgresSaver)")
    except Exception as e:
        logger.warning(f"Checkpointer 초기화 실패 (파이프라인 정상 동작, checkpoint 비활성화): {e}")
        _checkpointer = None
        _connection = None


def get_checkpointer():
    """현재 checkpointer 반환 (None 허용 — graceful degradation)"""
    return _checkpointer


async def cleanup_job_checkpoints(job_id: str) -> None:
    """특정 job(thread_id)의 모든 checkpoint 삭제"""
    if _connection is None:
        return

    try:
        async with _connection.cursor() as cur:
            # langgraph checkpoint 테이블에서 해당 thread_id의 데이터 삭제
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (job_id,),
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (job_id,),
            )
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (job_id,),
            )
        logger.info(f"[{job_id}] Checkpoint 정리 완료")
    except Exception as e:
        logger.warning(f"[{job_id}] Checkpoint 정리 실패 (무시): {e}")



async def close_checkpointer() -> None:
    """리소스 정리"""
    global _checkpointer, _connection

    if _connection is not None:
        try:
            await _connection.close()
            logger.info("Checkpointer 커넥션 종료 완료")
        except Exception as e:
            logger.warning(f"Checkpointer 커넥션 종료 실패: {e}")
    _checkpointer = None
    _connection = None
