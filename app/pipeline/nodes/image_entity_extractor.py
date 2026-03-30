"""image_entity_extractor_node: OpenAI Vision으로 이미지 설명 생성"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.db import get_app_setting_int
from app.models.state import PipelineState
from app.pipeline.external.llm_provider import parse_model_string, resolve_api_key
from app.pipeline.external.openai_client import describe_image
from app.utils.async_utils import create_rate_limit_handler, gather_with_backpressure

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 2000


def _build_page_text_map(elements: List[Dict[str, Any]]) -> Dict[int, str]:
    """elements에서 페이지별 텍스트 컨텍스트를 구성한다.

    같은 페이지의 type=="text" 요소를 리스트 순서대로 결합하고,
    MAX_CONTEXT_CHARS로 잘라낸다.
    """
    page_texts: Dict[int, List[str]] = {}
    for elem in elements:
        if elem.get("type") == "text":
            page = elem.get("page", 0)
            content = elem.get("content", "")
            if content:
                page_texts.setdefault(page, []).append(content)

    return {
        page: "\n".join(texts)[:MAX_CONTEXT_CHARS]
        for page, texts in page_texts.items()
    }


async def _process_single_image(
    index: int,
    page_num: Any,
    img_elem: Dict[str, Any],
    image_paths: List[str],
    api_key: str,
    language: str,
    page_text_map: Dict[int, str],
    job_id: str,
    total: int,
    on_rate_limit: Optional[Callable],
    model_str: str = "openai/gpt-4o",
) -> Dict[str, Any]:
    """단일 이미지에 대해 Vision API로 설명 생성 (병렬 실행 단위)"""
    page = img_elem.get("page", page_num)
    pos = img_elem.get("position", 0)

    # 매칭되는 이미지 파일 찾기
    matching_path = None
    for path in image_paths:
        if f"page_{page}_img_{pos}" in path:
            matching_path = path
            break

    if matching_path is None:
        return {
            "element_id": img_elem.get("id"),
            "page": page,
            "description": "",
            "error": "이미지 파일을 찾을 수 없습니다.",
        }

    try:
        page_context = page_text_map.get(page)
        description = await describe_image(
            matching_path, model_str=model_str, api_key=api_key, language=language,
            context=page_context, on_rate_limit=on_rate_limit,
        )
        logger.info(
            f"[{job_id}] 이미지 설명 생성 완료: page={page}, pos={pos}, context={'있음' if page_context else '없음'}"
        )
        return {
            "element_id": img_elem.get("id"),
            "page": page,
            "description": description,
        }
    except Exception as e:
        logger.warning(f"[{job_id}] 이미지 설명 생성 실패: {e}")
        return {
            "element_id": img_elem.get("id"),
            "page": page,
            "description": "",
            "error": str(e),
        }


async def image_entity_extractor_node(state: PipelineState, config: RunnableConfig) -> dict:
    """각 이미지에 대해 OpenAI Vision API로 설명 생성

    image_entities 키에만 기록 (병렬 브랜치 키 분리)
    """
    job_id = state["job_id"]
    # vision_model로 프로바이더 결정
    vision_model_str = config["configurable"].get("vision_model", settings.vision_model)
    provider, model = parse_model_string(vision_model_str)
    api_key = resolve_api_key(provider, config["configurable"])
    language = state.get("language", "Korean")
    image_paths = state.get("image_paths", [])
    page_elements = state.get("page_elements", {})
    elements = state.get("elements", [])
    page_text_map = _build_page_text_map(elements)

    # 모든 이미지 요소에 대해 설명 생성
    all_images = []
    for page_num, elems in page_elements.items():
        for img_elem in elems.get("images", []):
            all_images.append((page_num, img_elem))

    semaphore = asyncio.Semaphore(get_app_setting_int("entity_extractor_max_concurrency", default=3))
    pause_event = asyncio.Event()
    pause_event.set()
    on_rate_limit = create_rate_limit_handler(pause_event)

    coros = []
    for i, (page_num, img_elem) in enumerate(all_images):
        coros.append(_process_single_image(
            i, page_num, img_elem, image_paths, api_key, language,
            page_text_map, job_id, len(all_images), on_rate_limit,
            model_str=vision_model_str,
        ))

    image_entities = await gather_with_backpressure(semaphore, coros, pause_event)
    image_entities = [e for e in image_entities if e is not None]

    logger.info(f"[{job_id}] 이미지 엔티티 추출 완료: {len(image_entities)}개")

    return {"image_entities": image_entities}
