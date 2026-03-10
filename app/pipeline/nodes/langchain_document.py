"""langchain_document_node: enriched 요소로 LangChain Document 객체 생성 (.pkl)"""

import logging
import pickle
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.models.state import PipelineState
from app.services.file_manager import get_work_dir

logger = logging.getLogger(__name__)


def _elements_to_documents(
    elements: List[Dict[str, Any]],
    filename: str,
) -> List[Document]:
    """강화된 요소를 LangChain Document 객체로 변환"""
    documents: List[Document] = []

    for elem in elements:
        # 텍스트 내용 구성
        content_parts = []

        if elem["type"] == "text":
            content_parts.append(elem.get("content", ""))
        elif elem["type"] == "image":
            entity = elem.get("entity")
            if entity and entity.get("description"):
                content_parts.append(f"[Image Description]: {entity['description']}")
            else:
                content_parts.append("[Image]")
        elif elem["type"] == "table":
            entity = elem.get("entity")
            if entity and entity.get("structured_table"):
                content_parts.append(entity["structured_table"])
            else:
                content_parts.append(elem.get("content", ""))

        page_content = "\n".join(content_parts).strip()
        if not page_content:
            continue

        metadata = {
            "source": filename,
            "page": elem.get("page", 0),
            "type": elem.get("type", "text"),
            "element_id": elem.get("id"),
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


async def langchain_document_node(state: PipelineState) -> dict:
    """enriched 요소로 LangChain Document 객체 생성 및 pickle 저장"""
    job_id = state["job_id"]
    reconstructed = state.get("reconstructed_elements", [])

    # 원본 파일명 추출
    pdf_path = state.get("pdf_path", "")
    filename = pdf_path.split("/")[-1] if pdf_path else "unknown.pdf"
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename

    documents = _elements_to_documents(reconstructed, filename)

    work_dir = get_work_dir(job_id)
    pkl_path = work_dir / f"{job_id}_{base_name}.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(documents, f)

    logger.info(f"[{job_id}] LangChain Document 생성 완료: {len(documents)}개 → {pkl_path}")

    return {"pkl_path": str(pkl_path)}
