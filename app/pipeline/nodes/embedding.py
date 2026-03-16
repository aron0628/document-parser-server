"""embedding_node: LangChain Document 임베딩 생성 및 PostgreSQL 저장"""

import json
import logging
import pickle
from typing import List

from langchain_core.documents import Document
from openai import AsyncOpenAI

from app.config import settings
from app.db import get_pool
from app.models.state import PipelineState

logger = logging.getLogger(__name__)


def check_embedding(state: PipelineState) -> bool:
    """임베딩 활성화 여부 확인 (conditional_edges용)"""
    return state.get("enable_embedding", False)


async def embedding_node(state: PipelineState) -> dict:
    """pickle에서 Document 로드 후 Upstage 임베딩 생성 및 DB 저장"""
    job_id = state["job_id"]

    try:
        # pickle 로드
        pkl_path = state["pkl_path"]
        with open(pkl_path, "rb") as f:
            documents: List[Document] = pickle.load(f)

        if not documents:
            logger.info(f"[{job_id}] 임베딩할 Document 없음, 스킵")
            return {"embedding_count": 0}

        # Upstage 임베딩 클라이언트 (OpenAI 호환)
        client = AsyncOpenAI(
            api_key=state["upstage_api_key"],
            base_url="https://api.upstage.ai/v1",
        )
        embedding_model = state.get("embedding_model", settings.default_embedding_model)
        batch_size = settings.embedding_batch_size

        # 배치 임베딩 생성
        all_embeddings: list = []
        texts = [doc.page_content for doc in documents]

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
        for idx, (doc, embedding) in enumerate(zip(documents, all_embeddings)):
            records.append((
                job_id,
                idx,
                doc.metadata.get("page"),
                doc.metadata.get("category") or doc.metadata.get("type"),
                doc.page_content,
                json.dumps(doc.metadata, ensure_ascii=False),
                embedding,
            ))

        try:
            pool = get_pool()
            async with pool.connection() as conn:
                await conn.executemany(
                    """INSERT INTO document_embeddings
                       (job_id, element_index, page, element_type, content, metadata, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    records,
                )
                await conn.commit()
        except Exception as e:
            logger.error(f"[{job_id}] DB 임베딩 저장 실패: {e}")
            return {"embedding_count": 0}

        logger.info(f"[{job_id}] 임베딩 저장 완료: {len(documents)}개")
        return {"embedding_count": len(documents)}

    except Exception as e:
        logger.error(f"[{job_id}] embedding_node 실패: {e}")
        return {"embedding_count": 0}
