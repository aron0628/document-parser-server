"""table_entity_extractor_node: OpenAI로 테이블 구조화"""

import logging
from typing import Any, Dict, List

from app.models.state import PipelineState
from app.pipeline.external.openai_client import extract_table

logger = logging.getLogger(__name__)


async def table_entity_extractor_node(state: PipelineState) -> dict:
    """각 테이블에 대해 OpenAI API로 구조화된 데이터 추출

    table_entities 키에만 기록 (병렬 브랜치 키 분리)
    """
    job_id = state["job_id"]
    api_key = state["openai_api_key"]
    language = state.get("language", "Korean")
    page_elements = state.get("page_elements", {})

    table_entities: List[Dict[str, Any]] = []

    for page_num, elems in page_elements.items():
        for tbl_elem in elems.get("tables", []):
            table_content = tbl_elem.get("html", "") or tbl_elem.get("content", "")
            if not table_content:
                table_entities.append({
                    "element_id": tbl_elem.get("id"),
                    "page": tbl_elem.get("page", page_num),
                    "structured_table": "",
                    "error": "테이블 내용이 비어있습니다.",
                })
                continue

            try:
                structured = await extract_table(table_content, api_key, language)
                table_entities.append({
                    "element_id": tbl_elem.get("id"),
                    "page": tbl_elem.get("page", page_num),
                    "structured_table": structured,
                })
                logger.info(f"[{job_id}] 테이블 구조화 완료: page={page_num}")
            except Exception as e:
                logger.warning(f"[{job_id}] 테이블 구조화 실패: {e}")
                table_entities.append({
                    "element_id": tbl_elem.get("id"),
                    "page": tbl_elem.get("page", page_num),
                    "structured_table": "",
                    "error": str(e),
                })

    logger.info(f"[{job_id}] 테이블 엔티티 추출 완료: {len(table_entities)}개")

    return {"table_entities": table_entities}
