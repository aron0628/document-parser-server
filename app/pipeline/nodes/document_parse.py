"""document_parse_node: Upstage API로 현재 배치 PDF 파싱"""

import logging

from langchain_core.runnables import RunnableConfig

from app.models.state import PipelineState
from app.pipeline.external.upstage_client import parse_document

logger = logging.getLogger(__name__)


async def document_parse_node(state: PipelineState, config: RunnableConfig) -> dict:
    """현재 배치의 PDF 청크를 Upstage API로 파싱"""
    job_id = state["job_id"]
    current_idx = state.get("current_batch_index", 0)
    chunks = state.get("pdf_chunks", [])

    if current_idx >= len(chunks):
        return {}

    chunk_path = chunks[current_idx]
    api_key = config["configurable"]["upstage_api_key"]

    logger.info(f"[{job_id}] 배치 {current_idx+1}/{len(chunks)} 파싱 중: {chunk_path}")

    result = await parse_document(
        pdf_path=chunk_path,
        api_key=api_key,
    )

    # 기존 결과에 추가
    batch_results = list(state.get("batch_parse_results", []))
    batch_results.append(result)

    logger.info(f"[{job_id}] 배치 {current_idx+1}/{len(chunks)} 파싱 완료")

    return {
        "batch_parse_results": batch_results,
        "current_batch_index": current_idx + 1,
    }
