"""embedding_node: LangChain Document 임베딩 생성 및 PostgreSQL 저장"""

import json
import logging
import pickle
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from app.config import settings
from app.db import get_pool
from app.models.state import PipelineState

logger = logging.getLogger(__name__)


def _group_short_elements(
    documents: List[Document],
    chunk_size: int,
) -> List[Document]:
    """chunk_size//2 미만 element를 같은 page 내 인접끼리 그룹화한다.

    Tier 1 (짧은 element): 같은 page 내 인접 짧은 element를 하나의 Document로 합침
    Tier 2 (긴 element): 그대로 통과
    """
    threshold = chunk_size // 2
    grouped: List[Document] = []
    i = 0

    while i < len(documents):
        doc = documents[i]

        # Tier 2: 긴 element는 그대로 통과
        if len(doc.page_content) >= threshold:
            grouped.append(doc)
            i += 1
            continue

        # Tier 1: 같은 page 내 인접 짧은 element 수집
        page = doc.metadata.get("page", 0)
        group_contents = [doc.page_content]
        group_element_ids = [doc.metadata.get("element_id")]
        group_types = {doc.metadata.get("type", "text")}
        j = i + 1

        while j < len(documents):
            next_doc = documents[j]
            if (next_doc.metadata.get("page", 0) != page
                    or len(next_doc.page_content) >= threshold):
                break
            group_contents.append(next_doc.page_content)
            group_element_ids.append(next_doc.metadata.get("element_id"))
            group_types.add(next_doc.metadata.get("type", "text"))
            j += 1

        # "\n" 구분자 사용 ("\n\n" 아님 — RecursiveCharacterTextSplitter가
        # "\n\n"을 최우선 분할점으로 사용하므로 간섭 방지)
        merged_content = "\n".join(group_contents)
        merged_doc = Document(
            page_content=merged_content,
            metadata={
                "source": doc.metadata.get("source"),
                "page": page,
                "type": doc.metadata.get("type", "text"),
                "element_id": doc.metadata.get("element_id"),
                "grouped_element_ids": group_element_ids,
                "grouped_types": sorted(group_types),
                "is_grouped": len(group_contents) > 1,
            },
        )
        grouped.append(merged_doc)
        i = j

    return grouped


def _split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Two-tier 방식: 짧은 element 그룹화 후 split한다."""
    grouped = _group_short_elements(documents, chunk_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []
    for parent_idx, doc in enumerate(grouped):
        chunks = splitter.split_documents([doc])
        for chunk_idx, chunk in enumerate(chunks):
            chunk.metadata["parent_element_index"] = parent_idx
            chunk.metadata["chunk_index"] = chunk_idx
            all_chunks.append(chunk)

    return all_chunks


async def embedding_node(state: PipelineState, config: RunnableConfig) -> dict:
    """pickle에서 Document 로드 후 텍스트 분할, Upstage 임베딩 생성 및 DB 저장"""
    job_id = state["job_id"]

    try:
        # pickle 로드
        pkl_path = state["pkl_path"]
        with open(pkl_path, "rb") as f:
            documents: List[Document] = pickle.load(f)

        if not documents:
            logger.info(f"[{job_id}] 임베딩할 Document 없음, 스킵")
            return {"embedding_count": 0}

        # 텍스트 분할 (Two-tier: 짧은 element 그룹화 후 split)
        chunk_size = state.get("chunk_size", settings.chunk_size)
        chunk_overlap = state.get("chunk_overlap", settings.chunk_overlap)
        grouped = _group_short_elements(documents, chunk_size)
        chunks = _split_documents(documents, chunk_size, chunk_overlap)
        logger.info(
            f"[{job_id}] 텍스트 분할 완료: {len(documents)}개 element → "
            f"{len(grouped)}개 그룹 → {len(chunks)}개 chunk "
            f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )

        # Upstage 임베딩 클라이언트 (OpenAI 호환)
        client = AsyncOpenAI(
            api_key=config["configurable"]["upstage_api_key"],
            base_url="https://api.upstage.ai/v1",
        )
        embedding_model = state.get("embedding_model", settings.default_embedding_model)
        batch_size = settings.embedding_batch_size

        # 배치 임베딩 생성
        all_embeddings: list = []
        texts = [chunk.page_content for chunk in chunks]

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                response = await client.embeddings.create(
                    model=embedding_model,
                    input=batch_texts,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(
                    f"[{job_id}] 임베딩 배치 {i // batch_size + 1} 완료 "
                    f"({len(batch_embeddings)}개)"
                )
        except Exception as e:
            logger.error(f"[{job_id}] Upstage API 임베딩 실패: {e}")
            return {"embedding_count": 0}

        # DB INSERT
        records = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            records.append((
                job_id,
                idx,
                chunk.metadata.get("parent_element_index"),
                chunk.metadata.get("chunk_index", 0),
                chunk.metadata.get("page"),
                chunk.metadata.get("category") or chunk.metadata.get("type"),
                chunk.page_content,
                json.dumps(chunk.metadata, ensure_ascii=False),
                embedding,
            ))

        try:
            pool = get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """INSERT INTO document_embeddings
                           (job_id, element_index, parent_element_index, chunk_index,
                            page, element_type, content, metadata, embedding)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        records,
                    )
                await conn.commit()
        except Exception as e:
            logger.error(f"[{job_id}] DB 임베딩 저장 실패: {e}")
            return {"embedding_count": 0}

        logger.info(f"[{job_id}] 임베딩 저장 완료: {len(chunks)}개 chunk")
        return {"embedding_count": len(chunks)}

    except Exception as e:
        logger.error(f"[{job_id}] embedding_node 실패: {e}")
        return {"embedding_count": 0}
