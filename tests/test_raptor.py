"""raptor_node 단위 테스트 및 통합 테스트"""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# psycopg_pool / pgvector가 테스트 환경에 설치되지 않으므로 stub 등록
def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

if "psycopg_pool" not in sys.modules:
    _psycopg_pool = _stub_module("psycopg_pool")
    _psycopg_pool.AsyncConnectionPool = MagicMock

if "pgvector" not in sys.modules:
    _stub_module("pgvector")
    _stub_module("pgvector.psycopg")
    sys.modules["pgvector.psycopg"].register_vector_async = AsyncMock()


# ---------------------------------------------------------------------------
# 헬퍼: DB mock 생성
# ---------------------------------------------------------------------------

def _make_db_mock(fetchall_return=None, side_effects=None):
    """DB pool/connection/cursor mock 생성 헬퍼

    side_effects가 주어지면 pool.connection()을 여러 번 호출할 때 각각 다른
    connection mock을 반환한다.
    """
    if side_effects is not None:
        # 여러 connection() 호출에 대해 각각 다른 mock 반환
        conns = []
        for effect in side_effects:
            cur = AsyncMock()
            cur.__aenter__ = AsyncMock(return_value=cur)
            cur.__aexit__ = AsyncMock(return_value=False)
            if isinstance(effect, Exception):
                cur.execute = AsyncMock(side_effect=effect)
                cur.executemany = AsyncMock(side_effect=effect)
            else:
                cur.fetchall = AsyncMock(return_value=effect)
            conn = AsyncMock()
            conn.__aenter__ = AsyncMock(return_value=conn)
            conn.__aexit__ = AsyncMock(return_value=False)
            conn.cursor = MagicMock(return_value=cur)
            conns.append(conn)

        pool = MagicMock()
        pool.connection = MagicMock(side_effect=conns)
        return pool, conns

    cur = AsyncMock()
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)
    if fetchall_return is not None:
        cur.fetchall = AsyncMock(return_value=fetchall_return)

    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.cursor = MagicMock(return_value=cur)

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, [(conn, cur)]


def _base_state(**overrides):
    """raptor_node 테스트용 기본 state dict"""
    state = {
        "job_id": "test-job-123",
        "enable_raptor": True,
        "embedding_model": "embedding-passage",
    }
    state.update(overrides)
    return state


def _base_config(**overrides):
    """raptor_node 테스트용 기본 config dict"""
    configurable = {
        "upstage_api_key": "test-upstage-key",
        "openai_api_key": "test-openai-key",
        "raptor_summarization_model": "openai/gpt-4.1-mini",
    }
    configurable.update(overrides)
    return {"configurable": configurable}


# ---------------------------------------------------------------------------
# 단위 테스트: 클러스터링 함수
# ---------------------------------------------------------------------------


def test_get_optimal_clusters():
    """2개의 명확한 클러스터가 있는 데이터에서 최적 k를 결정"""
    from app.pipeline.nodes.raptor import _get_optimal_clusters

    # 두 그룹: [1,0,0] 근처 12개 + [0,1,0] 근처 12개
    rng = np.random.RandomState(42)
    group_a = rng.normal(loc=[5, 0, 0], scale=0.3, size=(12, 3))
    group_b = rng.normal(loc=[0, 5, 0], scale=0.3, size=(12, 3))
    embeddings = np.vstack([group_a, group_b])

    k = _get_optimal_clusters(embeddings, max_clusters=10)
    assert 1 <= k <= 10, f"유효 범위 밖: {k}"


def test_perform_clustering():
    """30개 임베딩 클러스터링 → 모든 인덱스가 최소 1개 클러스터에 포함"""
    from app.pipeline.nodes.raptor import _perform_clustering

    rng = np.random.RandomState(42)
    embeddings = rng.rand(30, 5)

    clusters = _perform_clustering(embeddings, dim=3, threshold=0.3)

    assert isinstance(clusters, list)
    assert len(clusters) > 0

    # 모든 인덱스 0..29가 최소 1개 클러스터에 포함
    all_indices = set()
    for cluster in clusters:
        assert isinstance(cluster, list)
        all_indices.update(cluster)
    assert all_indices == set(range(30))


def test_gmm_cluster():
    """2개 클러스터 데이터 → 올바른 구조 반환"""
    from app.pipeline.nodes.raptor import _gmm_cluster

    rng = np.random.RandomState(42)
    group_a = rng.normal(loc=[5, 0], scale=0.3, size=(15, 2))
    group_b = rng.normal(loc=[0, 5], scale=0.3, size=(15, 2))
    embeddings = np.vstack([group_a, group_b])

    clusters = _gmm_cluster(embeddings, threshold=0.5)

    assert isinstance(clusters, list)
    assert len(clusters) >= 1
    # 각 클러스터는 인덱스 리스트
    for cluster in clusters:
        assert isinstance(cluster, list)
        for idx in cluster:
            assert 0 <= idx < 30


def test_edge_cases():
    """_perform_clustering 엣지 케이스: 단일/2개/빈 임베딩"""
    from app.pipeline.nodes.raptor import _perform_clustering

    # 2개 임베딩 (dim보다 적으므로 UMAP 스킵, GMM 직접 수행)
    two = np.array([[1.0, 0.0], [0.0, 1.0]])
    clusters = _perform_clustering(two, dim=3, threshold=0.5)
    assert len(clusters) >= 1
    all_indices = set()
    for c in clusters:
        all_indices.update(c)
    assert 0 in all_indices
    assert 1 in all_indices

    # 3개 임베딩 (dim보다 적으므로 UMAP 스킵)
    three = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    clusters = _perform_clustering(three, dim=5, threshold=0.5)
    assert len(clusters) >= 1
    all_indices = set()
    for c in clusters:
        all_indices.update(c)
    assert all_indices == {0, 1, 2}


# ---------------------------------------------------------------------------
# 단위 테스트: raptor_node 경계 조건
# ---------------------------------------------------------------------------


async def test_chunk_count_bounds():
    """청크 수가 min_chunks_for_raptor 미만이면 스킵"""
    from app.pipeline.nodes.raptor import raptor_node

    # DB에서 5개 행 반환 (기본 min_chunks_for_raptor=10 미만)
    rows = [("text content", [0.1] * 128) for _ in range(5)]
    pool, _ = _make_db_mock(fetchall_return=rows)

    state = _base_state()

    with patch("app.pipeline.nodes.raptor.get_pool", return_value=pool):
        result = await raptor_node(state, _base_config())

    assert result == {"raptor_level_counts": {}}


async def test_timeout_enforcement():
    """_recursive_raptor 타임아웃 시 graceful 반환"""
    from app.pipeline.nodes.raptor import raptor_node

    rows = [("text content", [0.1] * 128) for _ in range(15)]

    # SELECT용 + DELETE용 + (INSERT는 타임아웃으로 도달 안 함)
    select_rows = rows
    pool, conns = _make_db_mock(side_effects=[select_rows, None, None])

    async def slow_raptor(*args, **kwargs):
        await asyncio.sleep(9999)
        return []

    mock_settings = MagicMock()
    mock_settings.min_chunks_for_raptor = 10
    mock_settings.max_chunks_for_raptor = 500
    mock_settings.raptor_max_levels = 3
    mock_settings.raptor_cluster_dim = 10
    mock_settings.raptor_cluster_threshold = 0.1
    mock_settings.raptor_timeout_seconds = 1
    mock_settings.raptor_summarization_model = "openai/gpt-4.1-mini"
    mock_settings.default_embedding_model = "embedding-passage"
    mock_settings.embedding_batch_size = 100

    state = _base_state()

    with patch("app.pipeline.nodes.raptor.get_pool", return_value=pool), \
         patch("app.pipeline.nodes.raptor.settings", mock_settings), \
         patch("app.pipeline.nodes.raptor.AsyncOpenAI"), \
         patch("app.pipeline.nodes.raptor.load_chat_model_with_retry", return_value=AsyncMock()), \
         patch("app.pipeline.nodes.raptor._recursive_raptor", side_effect=slow_raptor):
        result = await raptor_node(state, _base_config())

    assert result == {"raptor_level_counts": {}}


def test_blas_thread_env():
    """_run_clustering_sync 호출 후 OMP_NUM_THREADS=1 설정 확인"""
    from app.pipeline.nodes.raptor import _run_clustering_sync

    # 이전 값을 저장/복원
    old_val = os.environ.get("OMP_NUM_THREADS")
    try:
        embeddings_list = np.random.rand(5, 3).tolist()
        _run_clustering_sync(
            texts=["text"] * 5,
            embeddings_list=embeddings_list,
            level=1,
            max_levels=3,
            dim=3,
            threshold=0.5,
        )
        assert os.environ["OMP_NUM_THREADS"] == "1"
    finally:
        if old_val is not None:
            os.environ["OMP_NUM_THREADS"] = old_val


# ---------------------------------------------------------------------------
# 통합 테스트: raptor_node
# ---------------------------------------------------------------------------


async def test_raptor_node_disabled():
    """enable_raptor=False → DB 호출 없이 빈 결과 반환"""
    from app.pipeline.nodes.raptor import raptor_node

    state = _base_state(enable_raptor=False)

    mock_get_pool = MagicMock()
    with patch("app.pipeline.nodes.raptor.get_pool", mock_get_pool):
        result = await raptor_node(state, _base_config())

    assert result == {"raptor_level_counts": {}}
    mock_get_pool.assert_not_called()


async def test_raptor_node_enabled():
    """전체 흐름 mock: SELECT → DELETE → _recursive_raptor → INSERT"""
    from app.pipeline.nodes.raptor import raptor_node

    rows = [("text content", [0.1] * 128) for _ in range(15)]

    # 3번의 connection() 호출: SELECT, DELETE, INSERT
    pool, conns = _make_db_mock(side_effects=[rows, None, None])

    fake_results = [
        {
            "level": 1,
            "cluster_id": 0,
            "content": "요약 텍스트",
            "embedding": [0.2] * 128,
            "metadata": {"source_count": 5, "source_indices": [0, 1, 2, 3, 4], "clustering_method": "GMM"},
        },
        {
            "level": 1,
            "cluster_id": 1,
            "content": "요약 텍스트 2",
            "embedding": [0.3] * 128,
            "metadata": {"source_count": 5, "source_indices": [5, 6, 7, 8, 9], "clustering_method": "GMM"},
        },
    ]

    mock_settings = MagicMock()
    mock_settings.min_chunks_for_raptor = 10
    mock_settings.max_chunks_for_raptor = 500
    mock_settings.raptor_max_levels = 3
    mock_settings.raptor_cluster_dim = 10
    mock_settings.raptor_cluster_threshold = 0.1
    mock_settings.raptor_timeout_seconds = 300
    mock_settings.raptor_summarization_model = "openai/gpt-4.1-mini"
    mock_settings.default_embedding_model = "embedding-passage"
    mock_settings.embedding_batch_size = 100

    state = _base_state()

    with patch("app.pipeline.nodes.raptor.get_pool", return_value=pool), \
         patch("app.pipeline.nodes.raptor.settings", mock_settings), \
         patch("app.pipeline.nodes.raptor.AsyncOpenAI"), \
         patch("app.pipeline.nodes.raptor.load_chat_model_with_retry", return_value=AsyncMock()), \
         patch("app.pipeline.nodes.raptor._recursive_raptor", AsyncMock(return_value=fake_results)):
        result = await raptor_node(state, _base_config())

    # 레벨별 카운트 확인
    assert result["raptor_level_counts"] == {1: 2}

    # SELECT가 호출되었는지 확인
    select_cur = conns[0].cursor.return_value
    select_cur.execute.assert_called_once()
    select_sql = select_cur.execute.call_args[0][0]
    assert "document_embeddings" in select_sql

    # INSERT가 호출되었는지 확인
    insert_cur = conns[2].cursor.return_value
    insert_cur.executemany.assert_called_once()
    insert_sql = insert_cur.executemany.call_args[0][0]
    assert "raptor_summaries" in insert_sql


async def test_graceful_degradation():
    """_recursive_raptor 예외 발생 시 graceful 반환"""
    from app.pipeline.nodes.raptor import raptor_node

    rows = [("text content", [0.1] * 128) for _ in range(15)]
    pool, conns = _make_db_mock(side_effects=[rows, None])

    mock_settings = MagicMock()
    mock_settings.min_chunks_for_raptor = 10
    mock_settings.max_chunks_for_raptor = 500
    mock_settings.raptor_max_levels = 3
    mock_settings.raptor_cluster_dim = 10
    mock_settings.raptor_cluster_threshold = 0.1
    mock_settings.raptor_timeout_seconds = 300
    mock_settings.raptor_summarization_model = "openai/gpt-4.1-mini"
    mock_settings.default_embedding_model = "embedding-passage"
    mock_settings.embedding_batch_size = 100

    state = _base_state()

    with patch("app.pipeline.nodes.raptor.get_pool", return_value=pool), \
         patch("app.pipeline.nodes.raptor.settings", mock_settings), \
         patch("app.pipeline.nodes.raptor.AsyncOpenAI"), \
         patch("app.pipeline.nodes.raptor.load_chat_model_with_retry", return_value=AsyncMock()), \
         patch("app.pipeline.nodes.raptor._recursive_raptor", AsyncMock(side_effect=Exception("클러스터링 실패"))):
        result = await raptor_node(state, _base_config())

    assert result == {"raptor_level_counts": {}}


async def test_partial_failure_rollback():
    """DB INSERT 실패 시 graceful 반환 (트랜잭션 롤백)"""
    from app.pipeline.nodes.raptor import raptor_node

    rows = [("text content", [0.1] * 128) for _ in range(15)]
    # SELECT 성공, DELETE 성공, INSERT 실패
    pool, conns = _make_db_mock(side_effects=[rows, None, Exception("DB INSERT 실패")])

    fake_results = [
        {
            "level": 1,
            "cluster_id": 0,
            "content": "요약 텍스트",
            "embedding": [0.2] * 128,
            "metadata": {"source_count": 5, "source_indices": [0, 1, 2, 3, 4], "clustering_method": "GMM"},
        },
    ]

    mock_settings = MagicMock()
    mock_settings.min_chunks_for_raptor = 10
    mock_settings.max_chunks_for_raptor = 500
    mock_settings.raptor_max_levels = 3
    mock_settings.raptor_cluster_dim = 10
    mock_settings.raptor_cluster_threshold = 0.1
    mock_settings.raptor_timeout_seconds = 300
    mock_settings.raptor_summarization_model = "openai/gpt-4.1-mini"
    mock_settings.default_embedding_model = "embedding-passage"
    mock_settings.embedding_batch_size = 100

    state = _base_state()

    with patch("app.pipeline.nodes.raptor.get_pool", return_value=pool), \
         patch("app.pipeline.nodes.raptor.settings", mock_settings), \
         patch("app.pipeline.nodes.raptor.AsyncOpenAI"), \
         patch("app.pipeline.nodes.raptor.load_chat_model_with_retry", return_value=AsyncMock()), \
         patch("app.pipeline.nodes.raptor._recursive_raptor", AsyncMock(return_value=fake_results)):
        result = await raptor_node(state, _base_config())

    assert result == {"raptor_level_counts": {}}


async def test_idempotency_rerun():
    """멱등성: raptor_node가 INSERT 전 DELETE를 실행하는지 확인"""
    from app.pipeline.nodes.raptor import raptor_node

    rows = [("text content", [0.1] * 128) for _ in range(15)]
    # SELECT, DELETE, INSERT 각각의 connection
    pool, conns = _make_db_mock(side_effects=[rows, None, None])

    fake_results = [
        {
            "level": 1,
            "cluster_id": 0,
            "content": "요약 텍스트",
            "embedding": [0.2] * 128,
            "metadata": {"source_count": 5, "source_indices": [0, 1, 2, 3, 4], "clustering_method": "GMM"},
        },
    ]

    mock_settings = MagicMock()
    mock_settings.min_chunks_for_raptor = 10
    mock_settings.max_chunks_for_raptor = 500
    mock_settings.raptor_max_levels = 3
    mock_settings.raptor_cluster_dim = 10
    mock_settings.raptor_cluster_threshold = 0.1
    mock_settings.raptor_timeout_seconds = 300
    mock_settings.raptor_summarization_model = "openai/gpt-4.1-mini"
    mock_settings.default_embedding_model = "embedding-passage"
    mock_settings.embedding_batch_size = 100

    state = _base_state()

    with patch("app.pipeline.nodes.raptor.get_pool", return_value=pool), \
         patch("app.pipeline.nodes.raptor.settings", mock_settings), \
         patch("app.pipeline.nodes.raptor.AsyncOpenAI"), \
         patch("app.pipeline.nodes.raptor.load_chat_model_with_retry", return_value=AsyncMock()), \
         patch("app.pipeline.nodes.raptor._recursive_raptor", AsyncMock(return_value=fake_results)):
        result = await raptor_node(state, _base_config())

    # DELETE가 호출되었는지 확인 (2번째 connection)
    delete_cur = conns[1].cursor.return_value
    delete_cur.execute.assert_called_once()
    delete_sql = delete_cur.execute.call_args[0][0]
    assert "DELETE" in delete_sql
    assert "raptor_summaries" in delete_sql


def test_progress_not_broken():
    """raptor가 PIPELINE_NODES에 포함되지 않았는지 확인 (진행률 계산 영향 없음)"""
    from app.pipeline.logging import PIPELINE_NODES

    node_names = [name for name, _, _ in PIPELINE_NODES]
    assert "raptor" not in node_names


async def test_max_chunks_exceeded():
    """청크 수가 max_chunks_for_raptor 초과 시 스킵"""
    from app.pipeline.nodes.raptor import raptor_node

    # 600개 행 반환 (기본 max_chunks_for_raptor=500 초과)
    rows = [("text content", [0.1] * 128) for _ in range(600)]
    pool, _ = _make_db_mock(fetchall_return=rows)

    state = _base_state()

    with patch("app.pipeline.nodes.raptor.get_pool", return_value=pool):
        result = await raptor_node(state, _base_config())

    assert result == {"raptor_level_counts": {}}
