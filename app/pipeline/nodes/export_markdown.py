"""export_markdown node: un-enriched 요소로 Markdown 문서 생성"""

import logging
from typing import Any, Dict, List

from app.models.state import PipelineState
from app.services.file_manager import get_work_dir

logger = logging.getLogger(__name__)


def _elements_to_markdown(elements: List[Dict[str, Any]]) -> str:
    """요소 리스트를 Markdown 문서로 변환"""
    parts = []
    current_page = -1

    for elem in elements:
        page = elem.get("page", 0)
        if page != current_page:
            if current_page >= 0:
                parts.append("")
                parts.append("---")
                parts.append("")
            parts.append(f"## Page {page + 1}")
            parts.append("")
            current_page = page

        elem_type = elem.get("type", "text")
        if elem_type == "text":
            content = elem.get("content", "")
            if content.strip():
                parts.append(content)
                parts.append("")
        elif elem_type == "image":
            parts.append("![Image]()")
            parts.append("")
        elif elem_type == "table":
            content = elem.get("content", "")
            if content.strip():
                parts.append(content)
                parts.append("")

    return "\n".join(parts)


async def export_markdown_node(state: PipelineState) -> dict:
    """un-enriched 요소로 Markdown 파일 생성"""
    job_id = state["job_id"]
    elements = state.get("elements", [])

    work_dir = get_work_dir(job_id)
    md_path = work_dir / f"{job_id}_output.md"

    md_content = _elements_to_markdown(elements)
    md_path.write_text(md_content, encoding="utf-8")

    logger.info(f"[{job_id}] Markdown 내보내기 완료: {md_path}")

    return {"markdown_path": str(md_path)}
