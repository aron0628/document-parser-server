"""export_image node: PDF에서 이미지 영역을 추출하여 파일로 저장

include_image=False이면 image_paths를 빈 리스트로 반환 (no-op).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.models.state import PipelineState
from app.services.file_manager import get_work_dir

logger = logging.getLogger(__name__)


async def export_image_node(state: PipelineState) -> dict:
    """이미지 요소를 파일로 내보내기"""
    job_id = state["job_id"]
    include_image = state.get("include_image", True)
    elements = state.get("elements", [])

    if not include_image:
        logger.info(f"[{job_id}] include_image=False, 이미지 내보내기 건너뜀")
        return {"image_paths": []}

    image_dir = get_work_dir(job_id) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_elements = [e for e in elements if e["type"] == "image"]
    image_paths: List[str] = []

    for elem in image_elements:
        page = elem.get("page", 0)
        pos = elem.get("position", 0)
        # 실제 구현: PDF에서 bounding box 기반 이미지 크롭
        # 현재는 placeholder 이미지 생성
        img_filename = f"{job_id}_page_{page}_img_{pos}.png"
        img_path = image_dir / img_filename

        try:
            from PIL import Image
            # Placeholder: 빈 이미지 생성 (실제로는 PDF에서 크롭)
            img = Image.new("RGB", (200, 200), color="white")
            img.save(str(img_path))
            image_paths.append(str(img_path))
        except Exception as e:
            logger.warning(f"[{job_id}] 이미지 내보내기 실패 (page={page}): {e}")

    logger.info(f"[{job_id}] {len(image_paths)}개 이미지 내보내기 완료")

    return {"image_paths": image_paths}
