"""image_entity_extractor_node: OpenAI Vision으로 이미지 설명 생성"""

import logging
from typing import Any, Dict, List

from app.models.state import PipelineState
from app.pipeline.external.openai_client import describe_image

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


async def image_entity_extractor_node(state: PipelineState) -> dict:
    """각 이미지에 대해 OpenAI Vision API로 설명 생성

    image_entities 키에만 기록 (병렬 브랜치 키 분리)
    """
    job_id = state["job_id"]
    api_key = state["openai_api_key"]
    language = state.get("language", "Korean")
    image_paths = state.get("image_paths", [])
    page_elements = state.get("page_elements", {})
    elements = state.get("elements", [])
    page_text_map = _build_page_text_map(elements)

    image_entities: List[Dict[str, Any]] = []

    # 모든 이미지 요소에 대해 설명 생성
    all_images = []
    for page_num, elems in page_elements.items():
        for img_elem in elems.get("images", []):
            all_images.append((page_num, img_elem))

    for i, (page_num, img_elem) in enumerate(all_images):
        # 매칭되는 이미지 파일 찾기
        matching_path = None
        page = img_elem.get("page", page_num)
        pos = img_elem.get("position", 0)
        for path in image_paths:
            if f"page_{page}_img_{pos}" in path:
                matching_path = path
                break

        if matching_path is None:
            image_entities.append({
                "element_id": img_elem.get("id"),
                "page": page,
                "description": "",
                "error": "이미지 파일을 찾을 수 없습니다.",
            })
            continue

        try:
            page_context = page_text_map.get(page)
            description = await describe_image(matching_path, api_key, language, context=page_context)
            image_entities.append({
                "element_id": img_elem.get("id"),
                "page": page,
                "description": description,
            })
            logger.info(f"[{job_id}] 이미지 설명 생성 완료: page={page}, pos={pos}, context={'있음' if page_context else '없음'}")
        except Exception as e:
            logger.warning(f"[{job_id}] 이미지 설명 생성 실패: {e}")
            image_entities.append({
                "element_id": img_elem.get("id"),
                "page": page,
                "description": "",
                "error": str(e),
            })

    logger.info(f"[{job_id}] 이미지 엔티티 추출 완료: {len(image_entities)}개")

    return {"image_entities": image_entities}
