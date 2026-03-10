"""split_pdf_node: PDF를 batch_size 단위로 분할"""

import logging

from app.models.state import PipelineState
from app.services.file_manager import get_upload_dir
from app.utils.pdf_utils import split_pdf

logger = logging.getLogger(__name__)


async def split_pdf_node(state: PipelineState) -> dict:
    """PDF를 batch_size 페이지 단위로 분할하여 청크 목록 생성"""
    job_id = state["job_id"]
    pdf_path = state["pdf_path"]
    batch_size = state.get("batch_size", 30)
    test_page = state.get("test_page")

    output_dir = str(get_upload_dir(job_id) / "chunks")

    chunks = split_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        batch_size=batch_size,
        test_page=test_page,
    )

    logger.info(f"[{job_id}] PDF 분할 완료: {len(chunks)}개 청크")

    return {
        "pdf_chunks": chunks,
        "current_batch_index": 0,
        "batch_parse_results": [],
    }
