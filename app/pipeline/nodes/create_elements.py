"""create_elements_node: Upstage 파싱 결과에서 요소 객체 생성"""

import logging
from typing import Any, Dict, List

from app.models.state import PipelineState

logger = logging.getLogger(__name__)


def _parse_elements(merged_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Upstage API 결과를 표준 요소 객체 리스트로 변환"""
    elements: List[Dict[str, Any]] = []
    raw_elements = merged_result.get("elements", [])

    for idx, elem in enumerate(raw_elements):
        element = {
            "id": elem.get("id", idx),
            "type": elem.get("category", "text"),  # text, image, table
            "content": elem.get("content", {}).get("text", ""),
            "html": elem.get("content", {}).get("html", ""),
            "page": elem.get("page", 0),
            "position": idx,
            "bounding_box": elem.get("coordinates", {}),
        }

        # 이미지/테이블 타입 정규화
        category = element["type"].lower()
        if category in ("figure", "chart", "picture", "image"):
            element["type"] = "image"
            base64_encoding = elem.get("base64_encoding")
            if base64_encoding is not None:
                element["base64_encoding"] = base64_encoding
        elif category in ("table",):
            element["type"] = "table"
        else:
            element["type"] = "text"

        elements.append(element)

    return elements


def _strip_base64_from_merged(merged: Dict[str, Any]) -> Dict[str, Any]:
    """merged_parse_result에서 base64_encoding 필드를 제거한 shallow copy 반환 (원본 불변)"""
    stripped_elements = []
    for elem in merged.get("elements", []):
        stripped_elem = {k: v for k, v in elem.items() if k != "base64_encoding"}
        stripped_elements.append(stripped_elem)

    result = {**merged, "elements": stripped_elements}
    return result


async def create_elements_node(state: PipelineState) -> dict:
    """병합된 파싱 결과에서 요소 객체 리스트 생성"""
    job_id = state["job_id"]
    merged = state.get("merged_parse_result", {})

    elements = _parse_elements(merged)
    stripped_merged = _strip_base64_from_merged(merged)

    text_count = sum(1 for e in elements if e["type"] == "text")
    image_count = sum(1 for e in elements if e["type"] == "image")
    table_count = sum(1 for e in elements if e["type"] == "table")

    logger.info(
        f"[{job_id}] 요소 생성 완료: "
        f"text={text_count}, image={image_count}, table={table_count}"
    )

    return {"elements": elements, "merged_parse_result": stripped_merged}
