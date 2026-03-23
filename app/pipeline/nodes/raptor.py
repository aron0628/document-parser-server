"""raptor_node: RAPTOR 계층적 요약 임베딩 생성 및 PostgreSQL 저장

Recursive Abstractive Processing for Tree-Organized Retrieval (RAPTOR)
리프 임베딩을 클러스터링하고, 클러스터별 요약을 생성한 뒤,
요약 임베딩을 재귀적으로 반복하여 계층적 검색 구조를 구축한다.
"""

import asyncio
import json
import logging
import os
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from umap import UMAP

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.db import get_pool
from app.models.state import PipelineState
from app.utils.async_utils import gather_with_semaphore

logger = logging.getLogger(__name__)


def _global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int = 10,
    metric: str = "cosine",
) -> np.ndarray:
    """UMAP 글로벌 차원 축소"""
    n_neighbors = min(n_neighbors, len(embeddings) - 1)
    reducer = UMAP(
        n_components=dim,
        n_neighbors=max(n_neighbors, 2),
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    num_neighbors: int = 10,
    metric: str = "cosine",
) -> np.ndarray:
    """UMAP 로컬 차원 축소"""
    num_neighbors = min(num_neighbors, len(embeddings) - 1)
    reducer = UMAP(
        n_components=dim,
        n_neighbors=max(num_neighbors, 2),
        metric=metric,
        init="random",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = 42,
) -> int:
    """BIC 기반 최적 클러스터 수 결정 (GMM)"""
    max_k = min(max_clusters, len(embeddings))
    best_k = 1
    best_bic = float("inf")

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            random_state=random_state,
            covariance_type="full",
        )
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    return best_k


def _gmm_cluster(
    embeddings: np.ndarray,
    threshold: float = 0.5,
    random_state: int = 42,
) -> List[List[int]]:
    """GMM 클러스터링 (소프트 할당: 확률 threshold 이상이면 복수 클러스터 소속 가능)"""
    n_clusters = _get_optimal_clusters(embeddings, random_state=random_state)
    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=random_state,
        covariance_type="full",
    )
    gmm.fit(embeddings)
    probs = gmm.predict_proba(embeddings)

    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for idx in range(len(embeddings)):
        for cluster_id in range(n_clusters):
            if probs[idx, cluster_id] >= threshold:
                clusters[cluster_id].append(idx)

    # 빈 클러스터 제거
    return [c for c in clusters if c]


def _pca_reduce(embeddings: np.ndarray, target_dim: int = 50) -> np.ndarray:
    """UMAP 전 PCA 사전 차원축소 — 고차원 입력의 UMAP 성능 개선"""
    if embeddings.shape[1] <= target_dim:
        return embeddings
    n_components = min(target_dim, embeddings.shape[0], embeddings.shape[1])
    if n_components >= embeddings.shape[1]:
        return embeddings
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def _perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[List[int]]:
    """전체 클러스터링 파이프라인: 글로벌 UMAP → 글로벌 GMM → 로컬 UMAP → 로컬 GMM"""
    embeddings = _pca_reduce(embeddings)
    # UMAP spectral layout은 데이터가 적으면 eigsh 에러 발생 (k >= N)
    min_for_umap = max(dim + 2, 2 * dim)
    if len(embeddings) <= min_for_umap:
        # 데이터 부족 시 UMAP 스킵, 원본으로 GMM 직접 수행
        return _gmm_cluster(embeddings, threshold=threshold)

    # 글로벌 차원 축소 + GMM
    global_reduced = _global_cluster_embeddings(embeddings, dim)
    global_clusters = _gmm_cluster(global_reduced, threshold=threshold)

    # 각 글로벌 클러스터 내부에서 로컬 차원 축소 + GMM
    final_clusters: List[List[int]] = []
    for cluster_indices in global_clusters:
        if len(cluster_indices) <= min_for_umap:
            # 로컬 UMAP 수행 불가, 그대로 유지
            final_clusters.append(cluster_indices)
            continue

        cluster_embeddings = embeddings[cluster_indices]
        local_reduced = _local_cluster_embeddings(cluster_embeddings, dim)
        local_clusters = _gmm_cluster(local_reduced, threshold=threshold)

        # 로컬 인덱스를 원본 인덱스로 변환
        for local_cluster in local_clusters:
            original_indices = [cluster_indices[i] for i in local_cluster]
            final_clusters.append(original_indices)

    # 클러스터 수 상한 적용
    max_clusters = settings.raptor_max_clusters_per_level
    if len(final_clusters) > max_clusters:
        final_clusters.sort(key=len, reverse=True)
        logger.info(
            f"클러스터 수 {len(final_clusters)} → {max_clusters}개로 제한"
        )
        final_clusters = final_clusters[:max_clusters]

    return final_clusters


def _sanitize_text(text: str) -> str:
    """JSON 직렬화를 깨뜨리는 문자 제거

    - 제어 문자 (탭/개행/캐리지리턴 제외)
    - DEL (U+007F)
    - C1 제어 문자 (U+0080~U+009F) — PDF의 Windows-1252 인코딩에서 빈번
    - 서로게이트 문자 (U+D800~U+DFFF) — PDF 파싱 시 발생하는 깨진 유니코드
    - 기타 비표준 유니코드 (U+FFFE, U+FFFF)
    """
    # 서로게이트 및 깨진 유니코드 제거: UTF-8 왕복 인코딩
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    return "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r")
        or (32 <= ord(ch) < 0x7F)
        or (0x00A0 <= ord(ch) < 0xD800)
        or (0xDFFF < ord(ch) < 0xFFFE)
        or (ord(ch) > 0xFFFF)
    )


_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0


def _is_retryable(error: Exception) -> bool:
    """일시적 에러(재시도 가능) 여부 판별"""
    if isinstance(error, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(error, APIStatusError) and error.status_code >= 500:
        return True
    return False


async def _summarize_cluster(
    texts: List[str],
    client: AsyncOpenAI,
    model: str,
    max_chars: int = 100_000,
) -> Optional[str]:
    """클러스터 텍스트들을 LLM으로 요약.

    - 일시적 에러(429, 5xx, 네트워크): 지수 백오프 재시도
    - 영구적 에러(400 등): 즉시 스킵 (None 반환)
    """
    sanitized = [_sanitize_text(t) for t in texts]
    combined = "\n---\n".join(sanitized)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n... (truncated)"

    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "다음 텍스트들의 핵심 내용을 종합하여 하나의 상세한 요약을 작성하세요.",
                    },
                    {"role": "user", "content": combined},
                ],
            )
            content = response.choices[0].message.content
            return content if content else None
        except Exception as e:
            if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                wait = _INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"클러스터 요약 일시적 에러 (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"{wait}s 후 재시도: {e}"
                )
                await asyncio.sleep(wait)
                continue
            # 영구적 에러이거나 재시도 소진
            logger.warning(f"클러스터 요약 실패 (텍스트 {len(texts)}개): {e}")
            logger.debug(f"실패 클러스터 텍스트 샘플 (첫 200자): {combined[:200]!r}")
            return None
    return None


async def _embed_texts(
    texts: List[str],
    client: AsyncOpenAI,
    model: str,
    batch_size: int = 100,
) -> List[List[float]]:
    """Upstage API 배치 임베딩 생성"""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def _run_clustering_sync(
    texts: List[str],
    embeddings_list: List[List[float]],
    level: int,
    max_levels: int,
    dim: int,
    threshold: float,
) -> List[List[int]]:
    """CPU 바운드 클러스터링 동기 래퍼 (asyncio.to_thread에서 호출)"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    embeddings = np.array(embeddings_list)
    return _perform_clustering(embeddings, dim, threshold)


async def _recursive_raptor(
    texts: List[str],
    embeddings: np.ndarray,
    level: int,
    max_levels: int,
    dim: int,
    threshold: float,
    embed_client: AsyncOpenAI,
    embed_model: str,
    summarize_client: AsyncOpenAI,
    summarize_model: str,
    batch_size: int = 100,
    semaphore: Optional[asyncio.Semaphore] = None,
    deadline: Optional[float] = None,
    partial_results: Optional[List[Dict]] = None,
) -> List[Dict]:
    """재귀적 임베딩-클러스터링-요약 루프

    CPU 바운드 클러스터링은 asyncio.to_thread로, API 호출은 비동기로 수행한다.
    동시 API 호출 수는 semaphore로 제한하여 파일 디스크립터 고갈을 방지한다.
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(settings.raptor_max_concurrency)

    if partial_results is None:
        partial_results = []

    # deadline 초과 시 현재까지 결과로 중단
    if deadline and time.monotonic() > deadline:
        logger.warning(f"Level {level}: 시간 예산 소진, 재귀 중단")
        return partial_results

    # 기저 조건
    if level > max_levels:
        return partial_results
    if len(texts) < 3:
        return partial_results

    # 1. 클러스터링 (CPU-bound → thread pool)
    embeddings_list = embeddings.tolist()
    clusters = await asyncio.to_thread(
        _run_clustering_sync, texts, embeddings_list, level, max_levels, dim, threshold
    )

    if not clusters:
        return partial_results

    # 2. 클러스터별 요약 생성 (동시성 제한 병렬, 개별 실패 허용)
    summarize_coros = []
    for cluster_indices in clusters:
        cluster_texts = [texts[i] for i in cluster_indices]
        summarize_coros.append(
            _summarize_cluster(cluster_texts, summarize_client, summarize_model)
        )
    raw_summaries = await gather_with_semaphore(semaphore, summarize_coros)

    # 실패한 요약(None) 필터링
    valid = [
        (ci, s) for ci, s in zip(clusters, raw_summaries) if s is not None
    ]
    if not valid:
        logger.warning(f"Level {level}: 모든 클러스터 요약 실패, 재귀 중단")
        return partial_results

    valid_clusters, valid_summaries = zip(*valid)
    failed_count = len(clusters) - len(valid)
    failure_rate = failed_count / len(clusters)
    if failed_count > 0:
        logger.warning(
            f"Level {level}: {failed_count}/{len(clusters)}개 클러스터 요약 실패 "
            f"({failure_rate:.0%}), {len(valid)}개로 계속 진행"
        )
    # 실패율 50% 초과 시 RAPTOR 결과 품질이 무의미하므로 중단
    if failure_rate > 0.5:
        logger.error(
            f"Level {level}: 실패율 {failure_rate:.0%} > 50%, RAPTOR 재귀 중단"
        )
        return partial_results

    # 3. 요약 텍스트 임베딩 (비동기, 동시성 제한)
    async with semaphore:
        summary_embeddings = await _embed_texts(
            list(valid_summaries), embed_client, embed_model, batch_size
        )

    # 4. 현재 레벨 결과 수집
    for cluster_id, (cluster_indices, summary, embedding) in enumerate(
        zip(valid_clusters, valid_summaries, summary_embeddings)
    ):
        partial_results.append({
            "level": level,
            "cluster_id": cluster_id,
            "content": summary,
            "embedding": embedding,
            "metadata": {
                "source_count": len(cluster_indices),
                "source_indices": list(cluster_indices),
                "clustering_method": "GMM",
            },
        })

    # 5. 다음 레벨 재귀
    next_texts = list(valid_summaries)
    next_embeddings = np.array(summary_embeddings)
    await _recursive_raptor(
        next_texts,
        next_embeddings,
        level + 1,
        max_levels,
        dim,
        threshold,
        embed_client,
        embed_model,
        summarize_client,
        summarize_model,
        batch_size,
        semaphore,
        deadline,
        partial_results,
    )

    return partial_results


async def raptor_node(state: PipelineState, config: RunnableConfig) -> dict:
    """RAPTOR 계층적 요약 생성 및 DB 저장

    리프 임베딩을 DB에서 읽어 클러스터링 → 요약 → 임베딩을 재귀적으로 수행하고,
    결과를 raptor_summaries 테이블에 저장한다.
    실패 시 파이프라인을 중단하지 않고 빈 결과를 반환한다.
    """
    job_id = state["job_id"]

    try:
        # enable_raptor 플래그 확인 (conditional edge에서 이미 검사하지만 이중 확인)
        if not state.get("enable_raptor", False):
            logger.info(f"[{job_id}] RAPTOR 비활성화 상태, 스킵")
            return {"raptor_level_counts": {}}

        # DB에서 리프 임베딩 조회
        try:
            pool = get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT content, embedding FROM document_embeddings "
                        "WHERE job_id = %s ORDER BY element_index",
                        (job_id,),
                    )
                    rows = await cur.fetchall()
        except Exception as e:
            logger.error(f"[{job_id}] RAPTOR: 리프 임베딩 DB 조회 실패: {e}")
            return {"raptor_level_counts": {}}

        if not rows:
            logger.info(f"[{job_id}] RAPTOR: 리프 임베딩 없음, 스킵")
            return {"raptor_level_counts": {}}

        chunk_count = len(rows)
        logger.info(f"[{job_id}] RAPTOR: 리프 임베딩 {chunk_count}개 로드")

        # 청크 수 범위 확인
        if chunk_count < settings.min_chunks_for_raptor:
            logger.info(
                f"[{job_id}] RAPTOR: 청크 수({chunk_count})가 "
                f"최소 요구({settings.min_chunks_for_raptor}) 미만, 스킵"
            )
            return {"raptor_level_counts": {}}

        if chunk_count > settings.max_chunks_for_raptor:
            logger.info(
                f"[{job_id}] RAPTOR: 청크 수({chunk_count})가 "
                f"최대 한도({settings.max_chunks_for_raptor}) 초과, 스킵"
            )
            return {"raptor_level_counts": {}}

        # 텍스트 및 임베딩 분리 (vector 타입 미등록 커넥션은 문자열로 반환)
        texts = [row[0] for row in rows]
        embeddings_raw = []
        for row in rows:
            emb = row[1]
            if isinstance(emb, str):
                emb = json.loads(emb)
            embeddings_raw.append(emb)
        embeddings = np.array(embeddings_raw, dtype=np.float32)

        # 메모리 추정 (float32 기준)
        embedding_dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
        memory_bytes = chunk_count * embedding_dim * 4
        memory_mb = memory_bytes / (1024 * 1024)
        if memory_mb > 100:
            logger.warning(
                f"[{job_id}] RAPTOR: 예상 메모리 사용량 {memory_mb:.1f}MB (>100MB)"
            )
        logger.info(
            f"[{job_id}] RAPTOR: 임베딩 차원={embedding_dim}, "
            f"예상 메모리={memory_mb:.1f}MB"
        )

        # API 클라이언트 생성
        embed_client = AsyncOpenAI(
            api_key=config["configurable"]["upstage_api_key"],
            base_url="https://api.upstage.ai/v1",
        )
        summarize_client = AsyncOpenAI(
            api_key=config["configurable"]["openai_api_key"],
        )

        embed_model = state.get("embedding_model", settings.default_embedding_model)
        summarize_model = settings.raptor_summarization_model

        # 멱등성: 기존 결과 삭제
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM raptor_summaries WHERE job_id = %s",
                        (job_id,),
                    )
                await conn.commit()
        except Exception as e:
            logger.warning(f"[{job_id}] RAPTOR: 기존 결과 삭제 실패 (계속 진행): {e}")

        # 재귀적 RAPTOR 실행 (deadline 기반 시간 제한)
        deadline = time.monotonic() + settings.raptor_timeout_seconds
        results = await _recursive_raptor(
            texts=texts,
            embeddings=embeddings,
            level=1,
            max_levels=settings.raptor_max_levels,
            dim=settings.raptor_cluster_dim,
            threshold=settings.raptor_cluster_threshold,
            embed_client=embed_client,
            embed_model=embed_model,
            summarize_client=summarize_client,
            summarize_model=summarize_model,
            batch_size=settings.embedding_batch_size,
            deadline=deadline,
        )

        if not results:
            logger.info(f"[{job_id}] RAPTOR: 생성된 요약 없음")
            return {"raptor_level_counts": {}}

        # DB INSERT (단일 트랜잭션)
        try:
            records = []
            for r in results:
                records.append((
                    job_id,
                    r["level"],
                    r["cluster_id"],
                    r["content"],
                    json.dumps(r["metadata"], ensure_ascii=False),
                    r["embedding"],
                ))

            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """INSERT INTO raptor_summaries
                           (job_id, raptor_level, cluster_id, content, metadata, embedding)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        records,
                    )
                await conn.commit()
        except Exception as e:
            logger.error(f"[{job_id}] RAPTOR: DB 저장 실패: {e}")
            return {"raptor_level_counts": {}}

        # 레벨별 카운트 집계
        level_counts: Dict[int, int] = {}
        for r in results:
            lvl = r["level"]
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        total = sum(level_counts.values())
        logger.info(
            f"[{job_id}] RAPTOR 완료: 총 {total}개 요약 생성, "
            f"레벨별={level_counts}"
        )
        return {"raptor_level_counts": level_counts}

    except Exception as e:
        logger.error(
            f"[{job_id}] raptor_node 실패: {e}\n{traceback.format_exc()}"
        )
        return {"raptor_level_counts": {}}
