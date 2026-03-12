"""table_entity_extractor_node: OpenAI로 테이블 구조화"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.models.state import PipelineState
from app.pipeline.external.openai_client import extract_table
from app.pipeline.nodes.export_image import _FALLBACK_PNG

logger = logging.getLogger(__name__)


async def table_entity_extractor_node(state: PipelineState) -> dict:
    """각 테이블에 대해 OpenAI API로 구조화된 데이터 추출

    table_entities 키에만 기록 (병렬 브랜치 키 분리)
    """
    job_id = state["job_id"]
    api_key = state["openai_api_key"]
    language = state.get("language", "Korean")
    page_elements = state.get("page_elements", {})
    image_paths = state.get("image_paths", [])

    table_entities: List[Dict[str, Any]] = []

    for page_num, elems in page_elements.items():
        for tbl_elem in elems.get("tables", []):
            table_content = tbl_elem.get("html", "") or tbl_elem.get("content", "")

            # 매칭되는 테이블 이미지 파일 찾기
            matching_image = None
            page = tbl_elem.get("page", page_num)
            pos = tbl_elem.get("position", 0)
            for path in image_paths:
                if f"page_{page}_table_{pos}" in path:
                    matching_image = path
                    break

            # fallback PNG이면 이미지 없음으로 처리
            if matching_image is not None:
                if Path(matching_image).read_bytes() == _FALLBACK_PNG:
                    matching_image = None

            if not table_content and not matching_image:
                table_entities.append({
                    "element_id": tbl_elem.get("id"),
                    "page": tbl_elem.get("page", page_num),
                    "structured_table": "",
                    "error": "테이블 내용이 비어있습니다.",
                })
                continue

            try:
                structured = await extract_table(table_content, api_key, language, image_path=matching_image)
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
