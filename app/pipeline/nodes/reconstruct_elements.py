"""reconstruct_elements_node: 강화된 요소를 페이지 순서로 재구성"""

import logging
from typing import Any, Dict, List

from app.models.state import PipelineState

logger = logging.getLogger(__name__)


async def reconstruct_elements_node(state: PipelineState) -> dict:
    """병합된 요소를 (page, position) 순서로 정렬하여 문서 구조 재구성"""
    job_id = state["job_id"]
    merged_elements = state.get("merged_elements", [])

    # (page, position) 기준 정렬
    reconstructed = sorted(
        merged_elements,
        key=lambda e: (e.get("page", 0), e.get("position", 0)),
    )

    logger.info(f"[{job_id}] 문서 구조 재구성 완료: {len(reconstructed)}개 요소")

    return {"reconstructed_elements": reconstructed}
