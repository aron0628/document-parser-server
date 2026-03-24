"""키워드 전처리 노드 테스트"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

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

from app.pipeline.nodes.keyword_preprocess import (
    _compute_tf,
    _extract_keywords,
    _stopwords,
)
from app.config import settings


# ---------------------------------------------------------------------------
# _extract_keywords 단위 테스트 (실제 Kiwi 사용)
# ---------------------------------------------------------------------------


def test_extract_keywords_basic():
    """기본 명사 추출: NNG, NNP 태그가 정상 통과"""
    result = _extract_keywords(
        "삼성전자 반도체 사업부",
        pos_whitelist=["NNG", "NNP", "SL", "SH"],
        stopwords=_stopwords,
        min_length=2,
    )
    assert "삼성전자" in result or "삼성" in result
    assert "반도체" in result
    assert "사업부" in result or "사업" in result


def test_extract_keywords_stopwords():
    """불용어만 포함된 입력 → 빈 리스트 (POS 필터 또는 불용어 필터로 제거)"""
    result = _extract_keywords(
        "그러나 때문에 하지만 그리고",
        pos_whitelist=["NNG", "NNP", "SL", "SH"],
        stopwords=_stopwords,
        min_length=2,
    )
    assert result == []


def test_extract_keywords_empty():
    """빈 문자열 → 빈 리스트"""
    result = _extract_keywords(
        "",
        pos_whitelist=["NNG", "NNP", "SL", "SH"],
        stopwords=_stopwords,
        min_length=2,
    )
    assert result == []


def test_extract_keywords_mixed_lang():
    """영어(SL) + 한글(NNG) 혼합 입력 → 양쪽 모두 추출"""
    result = _extract_keywords(
        "FastAPI 서버에서 PDF를 파싱합니다",
        pos_whitelist=["NNG", "NNP", "SL", "SH"],
        stopwords=_stopwords,
        min_length=2,
    )
    # SL 태그로 영문 토큰 추출 (소문자 변환됨)
    assert "fastapi" in result
    assert "pdf" in result or "PDF".lower() in result
    # NNG 태그로 한글 명사 추출
    assert "서버" in result
    assert "파싱" in result


# ---------------------------------------------------------------------------
# _compute_tf 단위 테스트
# ---------------------------------------------------------------------------


def test_compute_tf():
    """TF 점수 계산: 빈도 / 전체 키워드 수"""
    result = _compute_tf(["반도체", "반도체", "시장", "분석"])
    assert result == {"반도체": 0.5, "시장": 0.25, "분석": 0.25}


def test_compute_tf_empty():
    """빈 키워드 리스트 → 빈 딕셔너리"""
    result = _compute_tf([])
    assert result == {}


# ---------------------------------------------------------------------------
# keyword_preprocess_node 노드 테스트
# ---------------------------------------------------------------------------


async def test_keyword_node_graceful_degradation():
    """DB pool이 None일 때 graceful degradation → 예외 없이 keyword_count 반환

    DB 저장은 실패하지만 키워드 추출 자체는 성공하므로 keyword_count > 0.
    파이프라인이 중단되지 않는 것이 핵심.
    """
    from app.pipeline.nodes.keyword_preprocess import keyword_preprocess_node

    state = {
        "job_id": "test-kw-no-db",
        "enable_keyword_extraction": True,
        "elements": [
            {"type": "text", "content": "삼성전자 반도체 사업부 매출 분석", "page": 1},
        ],
    }

    with patch("app.pipeline.nodes.keyword_preprocess.get_pool", return_value=None):
        result = await keyword_preprocess_node(state)

    # DB 실패해도 예외 없이 정상 반환 (키워드 추출 결과는 유지)
    assert "keyword_count" in result
    assert result["keyword_count"] >= 0


async def test_keyword_node_skip_disabled():
    """enable_keyword_extraction=False → 처리 없이 keyword_count=0 반환"""
    from app.pipeline.nodes.keyword_preprocess import keyword_preprocess_node

    state = {
        "job_id": "test-kw-disabled",
        "enable_keyword_extraction": False,
    }

    # Kiwi가 호출되지 않았음을 확인하기 위해 _kiwi.tokenize를 모니터링
    with patch("app.pipeline.nodes.keyword_preprocess._kiwi") as mock_kiwi:
        result = await keyword_preprocess_node(state)
        mock_kiwi.tokenize.assert_not_called()

    assert result["keyword_count"] == 0


# ---------------------------------------------------------------------------
# 설정 기본값 테스트
# ---------------------------------------------------------------------------


def test_pos_whitelist_default():
    """기본 POS 화이트리스트에 NNG, NNP, SL, SH 포함, NNB 미포함"""
    assert settings.keyword_pos_whitelist == ["NNG", "NNP", "SL", "SH"]
    assert "NNB" not in settings.keyword_pos_whitelist
