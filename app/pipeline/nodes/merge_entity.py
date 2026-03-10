"""merge_entity_node: AI 분석 결과를 원본 요소와 병합"""

import logging
from typing import Any, Dict, List

from app.models.state import PipelineState

logger = logging.getLogger(__name__)


async def merge_entity_node(state: PipelineState) -> dict:
    """이미지/테이블 엔티티 분석 결과를 원본 요소에 병합

    모든 요소에 entity 키 추가 (강화 데이터 또는 null)
    """
    job_id = state["job_id"]
    elements = state.get("elements", [])
    image_entities = state.get("image_entities", [])
    table_entities = state.get("table_entities", [])

    # element_id → entity 매핑 생성
    entity_map: Dict[Any, Dict[str, Any]] = {}
    for ie in image_entities:
        eid = ie.get("element_id")
        if eid is not None:
            entity_map[eid] = {
                "type": "image",
                "description": ie.get("description", ""),
            }
    for te in table_entities:
        eid = te.get("element_id")
        if eid is not None:
            entity_map[eid] = {
                "type": "table",
                "structured_table": te.get("structured_table", ""),
            }

    # 요소에 entity 키 추가
    merged: List[Dict[str, Any]] = []
    for elem in elements:
        merged_elem = dict(elem)
        eid = elem.get("id")
        merged_elem["entity"] = entity_map.get(eid, None)
        merged.append(merged_elem)

    enriched_count = sum(1 for e in merged if e["entity"] is not None)
    logger.info(f"[{job_id}] 엔티티 병합 완료: {enriched_count}/{len(merged)}개 강화됨")

    return {"merged_elements": merged}
