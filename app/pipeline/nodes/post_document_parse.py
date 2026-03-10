"""post_document_parse_node: 모든 배치 결과 병합 + 청크 정리"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from app.models.state import PipelineState

logger = logging.getLogger(__name__)


def _merge_parse_results(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """여러 배치의 Upstage API 결과를 하나로 병합"""
    if not batch_results:
        return {"elements": [], "pages": []}

    merged: Dict[str, Any] = {
        "elements": [],
        "pages": [],
    }

    page_offset = 0
    for batch in batch_results:
        elements = batch.get("elements", [])
        # 페이지 번호 오프셋 적용
        for elem in elements:
            elem_copy = dict(elem)
            if "page" in elem_copy:
                elem_copy["page"] = elem_copy["page"] + page_offset
            merged["elements"].append(elem_copy)

        pages = batch.get("pages", [])
        merged["pages"].extend(pages)
        page_offset += len(pages) if pages else 0

    return merged


async def post_document_parse_node(state: PipelineState) -> dict:
    """모든 배치 파싱 결과를 병합하고 분할 청크 파일 삭제"""
    job_id = state["job_id"]
    batch_results = state.get("batch_parse_results", [])

    logger.info(f"[{job_id}] {len(batch_results)}개 배치 결과 병합 중...")

    merged = _merge_parse_results(batch_results)

    logger.info(f"[{job_id}] 병합 완료: {len(merged.get('elements', []))}개 요소")

    # 분할 청크 파일 삭제 (더 이상 불필요)
    chunks = state.get("pdf_chunks", [])
    for chunk_path in chunks:
        chunk_dir = Path(chunk_path).parent
        if chunk_dir.name == "chunks" and chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
            logger.info(f"[{job_id}] 청크 디렉토리 정리: {chunk_dir}")
            break  # 같은 디렉토리이므로 한 번만

    return {"merged_parse_result": merged}
