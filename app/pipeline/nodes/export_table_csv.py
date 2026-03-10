"""export_table_csv node: un-enriched 요소에서 테이블을 CSV로 내보내기"""

import csv
import io
import logging
from typing import Any, Dict, List

from app.models.state import PipelineState
from app.services.file_manager import get_work_dir

logger = logging.getLogger(__name__)


async def export_table_csv_node(state: PipelineState) -> dict:
    """un-enriched 테이블 요소를 CSV 파일로 저장"""
    job_id = state["job_id"]
    elements = state.get("elements", [])

    work_dir = get_work_dir(job_id)
    csv_path = work_dir / f"{job_id}_tables.csv"

    table_elements = [e for e in elements if e["type"] == "table"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "table_index", "content"])
        for i, elem in enumerate(table_elements):
            writer.writerow([
                elem.get("page", 0),
                i,
                elem.get("content", ""),
            ])

    logger.info(f"[{job_id}] CSV 내보내기 완료: {len(table_elements)}개 테이블 → {csv_path}")

    return {"csv_path": str(csv_path)}
