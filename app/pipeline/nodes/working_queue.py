"""working_queue_node: 배치 큐 루프 컨트롤러"""

import logging

from app.models.state import PipelineState

logger = logging.getLogger(__name__)


async def working_queue_node(state: PipelineState) -> dict:
    """현재 배치 인덱스를 확인하고 상태만 반환 (분기는 check_queue가 담당)"""
    current = state.get("current_batch_index", 0)
    total = len(state.get("pdf_chunks", []))
    logger.debug(f"Working queue: batch {current+1}/{total}")
    return {}


def check_queue(state: PipelineState) -> bool:
    """배치 처리 계속 여부 판단 (conditional edge 함수)

    Returns:
        True: 아직 처리할 배치가 남아있음 → document_parse_node로
        False: 모든 배치 처리 완료 → post_document_parse_node로
    """
    current = state.get("current_batch_index", 0)
    total = len(state.get("pdf_chunks", []))
    return current < total
