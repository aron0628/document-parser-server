"""embedding_node: LangChain Document 임베딩 생성 및 PostgreSQL 저장"""

import json
import logging
import pickle
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from app.config import settings
from app.db import get_pool
from app.models.state import PipelineState

logger = logging.getLogger(__name__)


def _split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Document 리스트를 RecursiveCharacterTextSplitter로 분할하고 chunk 추적 메타데이터를 추가한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []
    for parent_idx, doc in enumerate(documents):
        chunks = splitter.split_documents([doc])
        for chunk_idx, chunk in enumerate(chunks):
            chunk.metadata["parent_element_index"] = parent_idx
            chunk.metadata["chunk_index"] = chunk_idx
            all_chunks.append(chunk)

    return all_chunks


async def embedding_node(state: PipelineState) -> dict:
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

        # 텍스트 분할
        chunk_size = state.get("chunk_size", settings.chunk_size)
        chunk_overlap = state.get("chunk_overlap", settings.chunk_overlap)
        chunks = _split_documents(documents, chunk_size, chunk_overlap)
        logger.info(
            f"[{job_id}] 텍스트 분할 완료: {len(documents)}개 Document → {len(chunks)}개 chunk "
            f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )

        # Upstage 임베딩 클라이언트 (OpenAI 호환)
        client = AsyncOpenAI(
            api_key=state["upstage_api_key"],
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
