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
# _group_short_elements 단위 테스트
# ---------------------------------------------------------------------------


def test_group_same_page():
    """같은 page의 짧은 element 3개 → 1개 그룹으로 병합"""
    from app.pipeline.nodes.embedding import _group_short_elements

    docs = [
        Document(page_content="짧은 텍스트 A", metadata={"page": 1, "type": "text", "element_id": "e1"}),
        Document(page_content="짧은 텍스트 B", metadata={"page": 1, "type": "text", "element_id": "e2"}),
        Document(page_content="짧은 텍스트 C", metadata={"page": 1, "type": "table", "element_id": "e3"}),
    ]

    grouped = _group_short_elements(docs, chunk_size=1000)

    assert len(grouped) == 1
    assert grouped[0].metadata["page"] == 1
    assert grouped[0].metadata["is_grouped"] is True
    assert grouped[0].metadata["grouped_element_ids"] == ["e1", "e2", "e3"]
    assert grouped[0].metadata["grouped_types"] == ["table", "text"]
    assert "짧은 텍스트 A\n짧은 텍스트 B\n짧은 텍스트 C" == grouped[0].page_content


def test_group_different_pages():
    """page 1에 2개 + page 2에 2개 → 그룹 2개"""
    from app.pipeline.nodes.embedding import _group_short_elements

    docs = [
        Document(page_content="A", metadata={"page": 1, "type": "text", "element_id": "e1"}),
        Document(page_content="B", metadata={"page": 1, "type": "text", "element_id": "e2"}),
        Document(page_content="C", metadata={"page": 2, "type": "text", "element_id": "e3"}),
        Document(page_content="D", metadata={"page": 2, "type": "text", "element_id": "e4"}),
    ]

    grouped = _group_short_elements(docs, chunk_size=1000)

    assert len(grouped) == 2
    assert grouped[0].metadata["page"] == 1
    assert grouped[0].metadata["grouped_element_ids"] == ["e1", "e2"]
    assert grouped[1].metadata["page"] == 2
    assert grouped[1].metadata["grouped_element_ids"] == ["e3", "e4"]


def test_group_mixed_tiers():
    """짧은(100자) + 긴(800자) + 짧은(100자), 같은 page → 3개 독립"""
    from app.pipeline.nodes.embedding import _group_short_elements

    short_text = "가" * 100
    long_text = "나" * 800

    docs = [
        Document(page_content=short_text, metadata={"page": 1, "type": "text", "element_id": "e1"}),
        Document(page_content=long_text, metadata={"page": 1, "type": "text", "element_id": "e2"}),
        Document(page_content=short_text, metadata={"page": 1, "type": "text", "element_id": "e3"}),
    ]

    # chunk_size=1000 → threshold=500. short(100) < 500, long(800) >= 500
    grouped = _group_short_elements(docs, chunk_size=1000)

    assert len(grouped) == 3
    # 첫 번째: 짧은 element 1개 (그룹 크기 1)
    assert grouped[0].metadata["is_grouped"] is False
    assert grouped[0].metadata["grouped_element_ids"] == ["e1"]
    # 두 번째: 긴 element 그대로 통과
    assert grouped[1].page_content == long_text
    # 세 번째: 짧은 element 1개 (긴 element가 끊음)
    assert grouped[2].metadata["is_grouped"] is False
    assert grouped[2].metadata["grouped_element_ids"] == ["e3"]


def test_group_adjacent_short_then_long():
    """짧은 + 짧은 + 긴 (같은 page) → 2개 (그룹 1 + 개별 1)"""
    from app.pipeline.nodes.embedding import _group_short_elements

    short_text = "가" * 100
    long_text = "나" * 800

    docs = [
        Document(page_content=short_text, metadata={"page": 1, "type": "text", "element_id": "e1"}),
        Document(page_content=short_text, metadata={"page": 1, "type": "image", "element_id": "e2"}),
        Document(page_content=long_text, metadata={"page": 1, "type": "text", "element_id": "e3"}),
    ]

    grouped = _group_short_elements(docs, chunk_size=1000)

    assert len(grouped) == 2
    assert grouped[0].metadata["is_grouped"] is True
    assert grouped[0].metadata["grouped_element_ids"] == ["e1", "e2"]
    assert grouped[0].metadata["grouped_types"] == ["image", "text"]
    assert grouped[1].page_content == long_text


def test_group_threshold_boundary():
    """content 길이가 정확히 threshold인 element → Tier 2 (개별 통과)"""
    from app.pipeline.nodes.embedding import _group_short_elements

    # chunk_size=1000 → threshold=500. 정확히 500자 → Tier 2
    exact_text = "가" * 500

    docs = [
        Document(page_content=exact_text, metadata={"page": 1, "type": "text", "element_id": "e1"}),
    ]

    grouped = _group_short_elements(docs, chunk_size=1000)

    assert len(grouped) == 1
    # Tier 2로 통과하므로 원본 Document 그대로 (grouped 메타데이터 없음)
    assert "is_grouped" not in grouped[0].metadata


def test_group_empty():
    """빈 리스트 → 빈 리스트 반환"""
    from app.pipeline.nodes.embedding import _group_short_elements

    grouped = _group_short_elements([], chunk_size=1000)
    assert grouped == []


# ---------------------------------------------------------------------------
# _split_documents 단위 테스트 (Two-tier)
# ---------------------------------------------------------------------------


def test_split_documents_basic():
    """긴 Document가 여러 chunk로 분할되고 metadata가 보존됨 (Tier 2 경로)"""
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


def test_split_documents_multiple_different_pages():
    """다른 page의 짧은 Document → 각각 별도 그룹, parent_element_index 올바르게 할당"""
    from app.pipeline.nodes.embedding import _split_documents

    docs = [
        Document(page_content="짧은 텍스트 A", metadata={"page": 1, "type": "text"}),
        Document(page_content="짧은 텍스트 B", metadata={"page": 2, "type": "text"}),
        Document(page_content="짧은 텍스트 C", metadata={"page": 3, "type": "text"}),
    ]

    chunks = _split_documents(docs, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["parent_element_index"] == i
        assert chunk.metadata["chunk_index"] == 0


def test_split_documents_grouped_then_split():
    """같은 page에 짧은 element 10개(각 200자) → 그룹 후 split"""
    from app.pipeline.nodes.embedding import _split_documents

    docs = [
        Document(
            page_content="가나다라마바사아자차 " * 20,  # ~220자
            metadata={"page": 1, "type": "text", "element_id": f"e{i}"},
        )
        for i in range(10)
    ]

    chunks = _split_documents(docs, chunk_size=1000, chunk_overlap=200)

    # 10개 element가 1개 그룹으로 합쳐진 뒤 (~2200자) split → 2개 이상 chunk
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["is_grouped"] is True


def test_split_documents_empty():
    """빈 Document 리스트 → 빈 리스트 반환"""
    from app.pipeline.nodes.embedding import _split_documents

    chunks = _split_documents([], chunk_size=1000, chunk_overlap=200)
    assert chunks == []


def test_split_documents_separator_no_interference():
    """그룹 내 '\\n' 구분자가 splitter의 '\\n\\n' 우선 분리와 충돌하지 않음"""
    from app.pipeline.nodes.embedding import _split_documents

    # 짧은 element 2개를 그룹화 → "\n"으로 합침 → "\n\n"이 없으므로 splitter가 그룹을 쪼개지 않음
    docs = [
        Document(page_content="첫 번째 문장입니다.", metadata={"page": 1, "type": "text", "element_id": "e1"}),
        Document(page_content="두 번째 문장입니다.", metadata={"page": 1, "type": "text", "element_id": "e2"}),
    ]

    chunks = _split_documents(docs, chunk_size=1000, chunk_overlap=200)

    # 합쳐도 chunk_size 미만이므로 1개 chunk
    assert len(chunks) == 1
    assert "첫 번째 문장입니다.\n두 번째 문장입니다." == chunks[0].page_content
    assert chunks[0].metadata["is_grouped"] is True


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

    # DB mock (pool.connection() → conn, conn.cursor() → cur)
    mock_cur = AsyncMock()
    mock_cur.__aenter__ = AsyncMock(return_value=mock_cur)
    mock_cur.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.cursor = MagicMock(return_value=mock_cur)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    state = {
        "job_id": "test-job-success",
        "pkl_path": str(pkl_file),
    }
    config = {"configurable": {"upstage_api_key": "test-key"}}

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state, config)

    # 짧은 텍스트이므로 chunk 수 = Document 수
    assert result["embedding_count"] == 2

    # DB INSERT에 9개 컬럼이 전달되었는지 확인
    call_args = mock_cur.executemany.call_args
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

    mock_cur = AsyncMock()
    mock_cur.__aenter__ = AsyncMock(return_value=mock_cur)
    mock_cur.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.cursor = MagicMock(return_value=mock_cur)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    state = {
        "job_id": "test-job-split",
        "pkl_path": str(pkl_file),
        "chunk_size": 500,
        "chunk_overlap": 100,
    }
    config = {"configurable": {"upstage_api_key": "test-key"}}

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state, config)

    # 1500자 텍스트 → chunk_size=500이므로 여러 chunk로 분할
    assert result["embedding_count"] > 1

    # DB에 저장된 레코드의 parent_element_index가 모두 0인지 확인 (원본 1개)
    call_args = mock_cur.executemany.call_args
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
    }
    config = {"configurable": {"upstage_api_key": "test-key"}}

    result = await embedding_node(state, config)
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
    }
    config = {"configurable": {"upstage_api_key": "test-key"}}

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client):
        result = await embedding_node(state, config)

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
    }
    config = {"configurable": {"upstage_api_key": "test-key"}}

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state, config)

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

    mock_cur = AsyncMock()
    mock_cur.__aenter__ = AsyncMock(return_value=mock_cur)
    mock_cur.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.cursor = MagicMock(return_value=mock_cur)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    state = {
        "job_id": "test-job-batch",
        "pkl_path": str(pkl_file),
    }
    config = {"configurable": {"upstage_api_key": "test-key"}}

    with patch("app.pipeline.nodes.embedding.AsyncOpenAI", return_value=mock_client), \
         patch("app.pipeline.nodes.embedding.get_pool", return_value=mock_pool):
        result = await embedding_node(state, config)

    # 짧은 텍스트이므로 splitting 후에도 150개 chunk
    assert result["embedding_count"] == 150
    assert len(call_args_list) == 2
    assert len(call_args_list[0]) == 100
    assert len(call_args_list[1]) == 50


# ---------------------------------------------------------------------------
# 통합 테스트
# ---------------------------------------------------------------------------


def test_build_graph():
    """compile_graph() 컴파일 성공 (노드/엣지 정상)"""
    from app.pipeline.graph import compile_graph

    compiled = compile_graph()
    assert compiled is not None

    # 핵심 노드 존재 확인
    node_names = set(compiled.get_graph().nodes.keys())
    assert "embedding" in node_names
    assert "langchain_document" in node_names


def test_graph_langchain_to_embedding():
    """langchain_document → embedding 직접 연결 확인"""
    from app.pipeline.graph import compile_graph

    compiled = compile_graph()
    graph_def = compiled.get_graph()

    edges = list(graph_def.edges)
    langchain_doc_edges = [e for e in edges if e[0] == "langchain_document"]

    targets = {e[1] for e in langchain_doc_edges}
    assert "embedding" in targets
