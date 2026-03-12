"""export_image node: 이미지 및 테이블 요소의 base64 데이터를 PNG 파일로 저장

include_image=False이면 image_paths를 빈 리스트로 반환 (no-op).
base64 데이터가 없거나 디코딩 실패 시 fallback PNG(1x1 투명)를 저장한다.
"""

import base64
import logging
from pathlib import Path
from typing import List

from app.models.state import PipelineState
from app.services.file_manager import get_work_dir

logger = logging.getLogger(__name__)

_FALLBACK_PNG = (
    b"\x89PNG\r\n\x1a\n"  # PNG signature
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def export_image_node(state: PipelineState) -> dict:
    """이미지 및 테이블 요소의 base64 데이터를 PNG 파일로 저장"""
    job_id = state["job_id"]
    include_image = state.get("include_image", True)
    elements = state.get("elements", [])

    if not include_image:
        logger.info(f"[{job_id}] include_image=False, 이미지 내보내기 건너뜀")
        return {"image_paths": []}

    image_dir = get_work_dir(job_id) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_elements = [e for e in elements if e["type"] in ("image", "table")]
    image_paths: List[str] = []

    for elem in image_elements:
        page = elem.get("page", 0)
        pos = elem.get("position", 0)
        type_prefix = "table" if elem["type"] == "table" else "img"
        img_filename = f"{job_id}_page_{page}_{type_prefix}_{pos}.png"
        img_path = image_dir / img_filename

        b64_data = elem.get("base64_encoding")
        if b64_data:
            try:
                img_path.write_bytes(base64.b64decode(b64_data))
                image_paths.append(str(img_path))
            except Exception as e:
                logger.warning(f"[{job_id}] base64 디코딩 실패, fallback PNG 저장 (page={page}): {e}")
                img_path.write_bytes(_FALLBACK_PNG)
                image_paths.append(str(img_path))
        else:
            logger.warning(f"[{job_id}] base64 데이터 없음, fallback PNG 저장 (page={page})")
            img_path.write_bytes(_FALLBACK_PNG)
            image_paths.append(str(img_path))

    logger.info(f"[{job_id}] {len(image_paths)}개 이미지 내보내기 완료")

    return {"image_paths": image_paths}
