"""page_elements_extractor_node: 요소를 페이지별로 이미지/테이블 분류"""

import logging
from typing import Any, Dict, List

from app.models.state import PipelineState

logger = logging.getLogger(__name__)


async def page_elements_extractor_node(state: PipelineState) -> dict:
    """요소를 페이지별로 그룹화하고 이미지/테이블로 분류"""
    job_id = state["job_id"]
    elements = state.get("elements", [])

    page_elements: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    for elem in elements:
        page = elem.get("page", 0)
        if page not in page_elements:
            page_elements[page] = {"images": [], "tables": []}

        if elem["type"] == "image":
            page_elements[page]["images"].append(elem)
        elif elem["type"] == "table":
            page_elements[page]["tables"].append(elem)

    total_images = sum(len(v["images"]) for v in page_elements.values())
    total_tables = sum(len(v["tables"]) for v in page_elements.values())

    logger.info(
        f"[{job_id}] 페이지별 분류: {len(page_elements)}페이지, "
        f"이미지={total_images}, 테이블={total_tables}"
    )

    return {"page_elements": page_elements}
