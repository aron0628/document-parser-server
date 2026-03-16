"""embedding 노드 단위 테스트 및 통합 테스트"""

import pickle
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

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
# _split_documents 단위 테스트
# ---------------------------------------------------------------------------


def test_split_documents_basic():
    """긴 Document가 여러 chunk로 분할되고 metadata가 보존됨"""
    from app.pipeline.nodes.embedding import _split_documents

    long_text = "가나다라마바사 " * 200  # 약 1600자
    docs = [Document(page_content=long_text, metadata={"page": 1, "type": "text", "element_id": "e1"})]

    chunks = _split_documents(docs, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["type"] == "text"
        assert chunk.metadata["parent_element_index"] == 0
        assert "chunk_index" in chunk.metadata


def test_split_documents_short_text():
    """chunk_size 미만의 짧은 Document는 분할되지 않음"""
    from app.pipeline.nodes.embedding import _split_documents

    docs = [Document(page_content="짧은 텍스트", metadata={"page": 1, "type": "text"})]

    chunks = _split_documents(docs, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0].metadata["parent_element_index"] == 0
    assert chunks[0].metadata["chunk_index"] == 0


def test_split_documents_multiple():
    """여러 Document의 parent_element_index가 올바르게 할당됨"""
    from app.pipeline.nodes.embedding import _split_documents

    docs = [
        Document(page_content="짧은 텍스트 A", metadata={"page": 1}),
        Document(page_content="짧은 텍스트 B", metadata={"page": 2}),
        Document(page_content="짧은 텍스트 C", metadata={"page": 3}),
    ]

    chunks = _split_documents(docs, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["parent_element_index"] == i
        assert chunk.metadata["chunk_index"] == 0


def test_split_documents_empty():
    """빈 Document 리스트 → 빈 리스트 반환"""
    from app.pipeline.nodes.embedding import _split_documents

    chunks = _split_documents([], chunk_size=1000, chunk_overlap=200)
    assert chunks == []


# ---------------------------------------------------------------------------
# embedding_node 단위 테스트
# ---------------------------------------------------------------------------


async def test_embedding_node_success(tmp_path):
    """mock Upstage API + mock DB → embedding_count 반환 (splitting 포함)"""
    from app.pipeline.nodes.embedding import embedding_node

    # pickle 파일 생성
    docs = [
        Document(page_content="텍스트 A", metadata={"page": 1, "category": "paragraph"}),
        Document(page_content="텍스트 B", metadata={"page": 2, "category": "table"}),
    ]
    pkl_file = tmp_path / "docs.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(docs, f)

    # API 응답 mock
    fake_embedding = [0.1] * 768
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding) for _ in docs]

    mock_embeddings = AsyncMock()
    mock_embeddings.create = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.embeddings = mock_embeddings

    # DB mock
    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    state = {
        "job_id": "test-job-success",
        "pkl_path": str(pkl_file),
        "upstage_api_key": "test-key",
        "enable_embedding": True,
    }

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state)

    # 짧은 텍스트이므로 chunk 수 = Document 수
    assert result["embedding_count"] == 2

    # DB INSERT에 9개 컬럼이 전달되었는지 확인
    call_args = mock_conn.executemany.call_args
    sql = call_args[0][0]
    assert "parent_element_index" in sql
    assert "chunk_index" in sql
    records = call_args[0][1]
    assert len(records) == 2
    # 각 레코드가 9개 필드인지 확인
    assert len(records[0]) == 9


async def test_embedding_node_with_splitting(tmp_path):
    """긴 Document가 splitting 후 chunk 단위로 임베딩/저장됨"""
    from app.pipeline.nodes.embedding import embedding_node

    long_text = "가나다라마바사아자차카타파하 " * 100  # 약 1500자
    docs = [
        Document(page_content=long_text, metadata={"page": 1, "type": "text"}),
    ]
    pkl_file = tmp_path / "docs.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(docs, f)

    fake_embedding = [0.1] * 768

    def make_response(texts):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding) for _ in texts]
        return mock_response

    async def fake_create(**kwargs):
        return make_response(kwargs["input"])

    mock_embeddings = AsyncMock()
    mock_embeddings.create = fake_create

    mock_client = MagicMock()
    mock_client.embeddings = mock_embeddings

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    state = {
        "job_id": "test-job-split",
        "pkl_path": str(pkl_file),
        "upstage_api_key": "test-key",
        "enable_embedding": True,
        "chunk_size": 500,
        "chunk_overlap": 100,
    }

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state)

    # 1500자 텍스트 → chunk_size=500이므로 여러 chunk로 분할
    assert result["embedding_count"] > 1

    # DB에 저장된 레코드의 parent_element_index가 모두 0인지 확인 (원본 1개)
    call_args = mock_conn.executemany.call_args
    records = call_args[0][1]
    for record in records:
        assert record[2] == 0  # parent_element_index


async def test_embedding_node_empty_documents(tmp_path):
    """빈 Document 리스트 pickle → embedding_count=0"""
    from app.pipeline.nodes.embedding import embedding_node

    pkl_file = tmp_path / "empty.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump([], f)

    state = {
        "job_id": "test-job-empty",
        "pkl_path": str(pkl_file),
        "upstage_api_key": "test-key",
        "enable_embedding": True,
    }

    result = await embedding_node(state)
    assert result["embedding_count"] == 0


async def test_embedding_node_api_error(tmp_path):
    """Upstage API 실패 시 graceful → embedding_count=0"""
    from app.pipeline.nodes.embedding import embedding_node

    docs = [Document(page_content="텍스트", metadata={"page": 1})]
    pkl_file = tmp_path / "docs.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(docs, f)

    mock_embeddings = AsyncMock()
    mock_embeddings.create = AsyncMock(side_effect=Exception("API 연결 실패"))

    mock_client = MagicMock()
    mock_client.embeddings = mock_embeddings

    state = {
        "job_id": "test-job-api-error",
        "pkl_path": str(pkl_file),
        "upstage_api_key": "test-key",
        "enable_embedding": True,
    }

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client):
        result = await embedding_node(state)

    assert result["embedding_count"] == 0


async def test_embedding_node_db_error(tmp_path):
    """DB 연결 실패 시 graceful → embedding_count=0"""
    from app.pipeline.nodes.embedding import embedding_node

    docs = [Document(page_content="텍스트", metadata={"page": 1})]
    pkl_file = tmp_path / "docs.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(docs, f)

    fake_embedding = [0.1] * 768
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding)]

    mock_embeddings = AsyncMock()
    mock_embeddings.create = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.embeddings = mock_embeddings

    # DB pool이 connection() 호출 시 예외 발생
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=Exception("DB 연결 실패"))

    state = {
        "job_id": "test-job-db-error",
        "pkl_path": str(pkl_file),
        "upstage_api_key": "test-key",
        "enable_embedding": True,
    }

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state)

    assert result["embedding_count"] == 0


async def test_embedding_node_batch_split(tmp_path):
    """150개 Document → 배치 크기(100) 기준으로 2번 배치 호출 확인"""
    from app.pipeline.nodes.embedding import embedding_node

    docs = [
        Document(page_content=f"텍스트 {i}", metadata={"page": i % 10 + 1})
        for i in range(150)
    ]
    pkl_file = tmp_path / "large.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(docs, f)

    fake_embedding = [0.1] * 768

    def make_response(texts):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding) for _ in texts]
        return mock_response

    call_args_list = []

    async def fake_create(**kwargs):
        call_args_list.append(kwargs["input"])
        return make_response(kwargs["input"])

    mock_embeddings = AsyncMock()
    mock_embeddings.create = fake_create

    mock_client = MagicMock()
    mock_client.embeddings = mock_embeddings

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    state = {
        "job_id": "test-job-batch",
        "pkl_path": str(pkl_file),
        "upstage_api_key": "test-key",
        "enable_embedding": True,
    }

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state)

    # 짧은 텍스트이므로 splitting 후에도 150개 chunk
    assert result["embedding_count"] == 150
    assert len(call_args_list) == 2
    assert len(call_args_list[0]) == 100
    assert len(call_args_list[1]) == 50


# ---------------------------------------------------------------------------
# 통합 테스트
# ---------------------------------------------------------------------------


def test_build_graph():
    """build_graph() 컴파일 성공 (노드/엣지 정상)"""
    from app.pipeline.graph import build_graph

    compiled = build_graph()
    assert compiled is not None

    # 핵심 노드 존재 확인
    node_names = set(compiled.get_graph().nodes.keys())
    assert "embedding" in node_names
    assert "langchain_document" in node_names


def test_graph_langchain_to_embedding():
    """langchain_document → embedding 직접 연결 확인"""
    from app.pipeline.graph import build_graph

    compiled = build_graph()
    graph_def = compiled.get_graph()

    edges = list(graph_def.edges)
    langchain_doc_edges = [e for e in edges if e[0] == "langchain_document"]

    targets = {e[1] for e in langchain_doc_edges}
    assert "embedding" in targets
