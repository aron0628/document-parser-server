"""export_html node: un-enriched 요소로 HTML 문서 생성"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.models.state import PipelineState
from app.services.file_manager import get_work_dir

logger = logging.getLogger(__name__)


def _elements_to_html(elements: List[Dict[str, Any]], title: str = "Document") -> str:
    """요소 리스트를 HTML 문서로 변환"""
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title></head><body>",
    ]

    current_page = -1
    for elem in elements:
        page = elem.get("page", 0)
        if page != current_page:
            if current_page >= 0:
                parts.append("</div>")
            parts.append(f"<div class='page' data-page='{page}'>")
            parts.append(f"<h2>Page {page + 1}</h2>")
            current_page = page

        elem_type = elem.get("type", "text")
        if elem_type == "text":
            content = elem.get("content", "")
            if content.strip():
                parts.append(f"<p>{content}</p>")
        elif elem_type == "image":
            parts.append("<div class='image'>[Image]</div>")
        elif elem_type == "table":
            html_content = elem.get("html", "")
            if html_content:
                parts.append(f"<div class='table'>{html_content}</div>")
            else:
                parts.append(f"<div class='table'><pre>{elem.get('content', '')}</pre></div>")

    if current_page >= 0:
        parts.append("</div>")
    parts.append("</body></html>")

    return "\n".join(parts)


async def export_html_node(state: PipelineState) -> dict:
    """un-enriched 요소로 HTML 파일 생성"""
    job_id = state["job_id"]
    elements = state.get("elements", [])

    work_dir = get_work_dir(job_id)
    html_path = work_dir / f"{job_id}_output.html"

    html_content = _elements_to_html(elements, title=f"Document {job_id}")
    html_path.write_text(html_content, encoding="utf-8")

    logger.info(f"[{job_id}] HTML 내보내기 완료: {html_path}")

    return {"html_path": str(html_path)}
