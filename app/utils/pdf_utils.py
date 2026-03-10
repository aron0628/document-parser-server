"""PDF 분할 유틸리티"""

import logging
from pathlib import Path
from typing import List, Optional

from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def split_pdf(
    pdf_path: str,
    output_dir: str,
    batch_size: int = 30,
    test_page: Optional[int] = None,
) -> List[str]:
    """PDF를 batch_size 페이지 단위로 분할

    Args:
        pdf_path: 원본 PDF 파일 경로
        output_dir: 분할 파일 저장 디렉토리
        batch_size: 배치당 페이지 수
        test_page: 처리할 최대 페이지 수 (None이면 전체)

    Returns:
        분할된 PDF 파일 경로 목록
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # test_page가 설정되면 해당 페이지까지만 처리
    if test_page is not None:
        total_pages = min(total_pages, test_page)

    logger.info(f"PDF 분할: 총 {total_pages}페이지, batch_size={batch_size}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    chunks: List[str] = []
    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)
        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])

        chunk_path = output_dir_path / f"chunk_{start:04d}_{end:04d}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)

        chunks.append(str(chunk_path))
        logger.info(f"  청크 생성: 페이지 {start+1}-{end} → {chunk_path.name}")

    return chunks
