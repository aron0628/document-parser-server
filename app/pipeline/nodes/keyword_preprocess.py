"""keyword_preprocess_node: 문서 요소에서 키워드 추출 및 TF 점수 계산"""

import json
import logging
from collections import Counter
from pathlib import Path

from app.config import settings
from app.db import get_app_setting_bool, get_pool
from app.models.state import PipelineState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 모듈-레벨 싱글턴 (최초 import 시 1회 로드)
# ---------------------------------------------------------------------------


def _load_stopwords() -> set[str]:
    """불용어 사전을 파일에서 로드한다."""
    path = Path(__file__).parent.parent.parent / "resources" / "korean_stopwords.txt"
    try:
        with open(path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning(f"불용어 파일을 찾을 수 없음: {path}")
        return set()


_stopwords: set[str] = _load_stopwords()

try:
    from kiwipiepy import Kiwi

    _kiwi = Kiwi(num_workers=settings.kiwi_num_workers)
except Exception as e:
    logger.warning(f"Kiwi 초기화 실패 (키워드 추출 비활성화): {e}")
    _kiwi = None

# ---------------------------------------------------------------------------
# 순수 로직 함수
# ---------------------------------------------------------------------------


def _extract_keywords(
    text: str,
    pos_whitelist: list[str],
    stopwords: set[str],
    min_length: int,
) -> list[str]:
    """Kiwi 형태소 분석 후 POS 필터 + 불용어/길이 필터를 거쳐 키워드 리스트를 반환한다."""
    # Stage 1: 형태소 분석 → POS 화이트리스트 필터
    tokens = _kiwi.tokenize(text)
    candidates = [token.form for token in tokens if token.tag in pos_whitelist]

    # Stage 2: 불용어 제거 + 최소 길이 필터
    keywords = [
        w.lower()
        for w in candidates
        if len(w) >= min_length and w.lower() not in stopwords
    ]
    return keywords


def _compute_tf(keywords: list[str]) -> dict[str, float]:
    """키워드 빈도(TF) 점수를 계산한다."""
    if not keywords:
        return {}
    counts = Counter(keywords)
    total = len(keywords)
    return {word: count / total for word, count in counts.items()}


# ---------------------------------------------------------------------------
# 파이프라인 노드
# ---------------------------------------------------------------------------


async def keyword_preprocess_node(state: PipelineState) -> dict:
    """문서 요소에서 키워드를 추출하고 TF 점수를 DB에 저장한다.

    Triple try/except 패턴:
      - Outer: Kiwi 초기화 실패
      - Middle: 키워드 추출 실패
      - Inner: DB 저장 실패
    항상 {"keyword_count": N}을 반환하며 파이프라인을 중단하지 않는다.
    """
    job_id = state["job_id"]

    if not state.get("enable_keyword_extraction", get_app_setting_bool("enable_keyword_extraction", default=True)):
        logger.info(f"[{job_id}] 키워드 추출 비활성화, 스킵")
        return {"keyword_count": 0}

    try:
        if _kiwi is None:
            logger.warning(f"[{job_id}] Kiwi 미초기화, 키워드 추출 스킵")
            return {"keyword_count": 0}

        elements = state.get("elements", [])
        if not elements:
            logger.info(f"[{job_id}] 요소 없음, 키워드 추출 스킵")
            return {"keyword_count": 0}

        pos_whitelist = settings.keyword_pos_whitelist
        min_length = settings.keyword_min_length
        records = []

        try:
            for idx, elem in enumerate(elements):
                elem_type = elem.get("type", "")
                content = elem.get("content", "")

                if elem_type not in ("text", "table") or not content:
                    continue

                keywords = _extract_keywords(
                    content, pos_whitelist, _stopwords, min_length
                )
                if not keywords:
                    continue

                tf_scores = _compute_tf(keywords)
                records.append((
                    job_id,
                    idx,
                    elem.get("page", 0),
                    keywords,  # psycopg가 list를 PostgreSQL TEXT[]로 자동 변환
                    json.dumps(tf_scores, ensure_ascii=False),
                    len(keywords),
                ))
        except Exception as e:
            logger.error(f"[{job_id}] 키워드 추출 실패: {e}")
            return {"keyword_count": 0}

        keyword_count = len(records)

        if not records:
            logger.info(f"[{job_id}] 추출된 키워드 없음")
            return {"keyword_count": 0}

        # DB 저장
        try:
            pool = get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """INSERT INTO document_keywords
                           (job_id, element_index, page, keywords, tf_scores, keyword_count)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        records,
                    )
                await conn.commit()
            logger.info(f"[{job_id}] 키워드 저장 완료: {keyword_count}개 요소")
        except Exception as e:
            logger.warning(f"[{job_id}] DB 키워드 저장 실패 (결과는 유지): {e}")

        return {"keyword_count": keyword_count}

    except Exception as e:
        logger.error(f"[{job_id}] keyword_preprocess_node 실패: {e}")
        return {"keyword_count": 0}
