"""table_entity_extractor_node: OpenAI로 테이블 구조화"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.db import get_app_setting_int
from app.models.state import PipelineState
from app.pipeline.external.llm_provider import parse_model_string, resolve_api_key
from app.pipeline.external.openai_client import extract_table
from app.pipeline.nodes.export_image import _FALLBACK_PNG
from app.utils.async_utils import create_rate_limit_handler, gather_with_backpressure

logger = logging.getLogger(__name__)


async def _process_single_table(
    index: int,
    page_num: Any,
    tbl_elem: Dict[str, Any],
    image_paths: List[str],
    api_key: str,
    language: str,
    job_id: str,
    total: int,
    on_rate_limit: Optional[Callable],
    model_str: str = "openai/gpt-4o",
) -> Dict[str, Any]:
    """단일 테이블 엔티티 추출 (에러 시 graceful degradation)"""
    table_content = tbl_elem.get("html", "") or tbl_elem.get("content", "")

    # 매칭되는 테이블 이미지 파일 찾기
    matching_image = None
    page = tbl_elem.get("page", page_num)
    pos = tbl_elem.get("position", 0)
    for path in image_paths:
        if f"page_{page}_table_{pos}" in path:
            matching_image = path
            break

    # fallback PNG이면 이미지 없음으로 처리
    if matching_image is not None:
        if Path(matching_image).read_bytes() == _FALLBACK_PNG:
            matching_image = None

    if not table_content and not matching_image:
        return {
            "element_id": tbl_elem.get("id"),
            "page": tbl_elem.get("page", page_num),
            "structured_table": "",
            "error": "테이블 내용이 비어있습니다.",
        }

    try:
        structured = await extract_table(
            table_content, model_str=model_str, api_key=api_key, language=language,
            image_path=matching_image, on_rate_limit=on_rate_limit,
        )
        logger.info(f"[{job_id}] 테이블 구조화 완료: page={page_num}")
        return {
            "element_id": tbl_elem.get("id"),
            "page": tbl_elem.get("page", page_num),
            "structured_table": structured,
        }
    except Exception as e:
        logger.warning(f"[{job_id}] 테이블 구조화 실패: {e}")
        return {
            "element_id": tbl_elem.get("id"),
            "page": tbl_elem.get("page", page_num),
            "structured_table": "",
            "error": str(e),
        }


async def table_entity_extractor_node(state: PipelineState, config: RunnableConfig) -> dict:
    """각 테이블에 대해 OpenAI API로 구조화된 데이터 추출

    table_entities 키에만 기록 (병렬 브랜치 키 분리)
    """
    job_id = state["job_id"]
    # vision_model로 프로바이더 결정
    vision_model_str = config["configurable"].get("vision_model", settings.vision_model)
    provider, model = parse_model_string(vision_model_str)
    api_key = resolve_api_key(provider, config["configurable"])
    language = state.get("language", "Korean")
    page_elements = state.get("page_elements", {})
    image_paths = state.get("image_paths", [])

    # Flatten all tables across pages
    all_tables = []
    for page_num, elems in page_elements.items():
        for tbl_elem in elems.get("tables", []):
            all_tables.append((page_num, tbl_elem))

    semaphore = asyncio.Semaphore(get_app_setting_int("entity_extractor_max_concurrency", default=3))
    pause_event = asyncio.Event()
    pause_event.set()
    on_rate_limit = create_rate_limit_handler(pause_event)

    coros = []
    for i, (page_num, tbl_elem) in enumerate(all_tables):
        coros.append(_process_single_table(
            i, page_num, tbl_elem, image_paths, api_key, language,
            job_id, len(all_tables), on_rate_limit,
            model_str=vision_model_str,
        ))

    table_entities = await gather_with_backpressure(semaphore, coros, pause_event)
    # Filter out None results if any
    table_entities = [e for e in table_entities if e is not None]

    logger.info(f"[{job_id}] 테이블 엔티티 추출 완료: {len(table_entities)}개")

    return {"table_entities": table_entities}
