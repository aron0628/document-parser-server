"""ZIP 패키징 유틸리티

ZIP 내부 구조: {job_id}/{job_id}_{filename}.ext (이중 중첩)
"""

import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.services.file_manager import get_result_dir

logger = logging.getLogger(__name__)


def create_result_zip(
    job_id: str,
    original_filename: str,
    html_path: Optional[str] = None,
    markdown_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    pkl_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
) -> str:
    """결과 파일들을 ZIP으로 패키징

    ZIP 내부 구조:
        {job_id}/
        ├── {job_id}_{base_name}.html
        ├── {job_id}_{base_name}.md
        ├── {job_id}_{base_name}_tables.csv
        ├── {job_id}_{base_name}.pkl
        └── images/
            ├── {job_id}_page_0_img_0.png
            └── ...

    Returns:
        "result/{job_id}_{timestamp}.zip" 형식의 zip_filename
    """
    base_name = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_dir = get_result_dir()
    zip_name = f"{job_id}_{timestamp}.zip"
    zip_full_path = result_dir / zip_name

    with zipfile.ZipFile(zip_full_path, "w", zipfile.ZIP_DEFLATED) as zf:
        prefix = job_id  # ZIP 내부 최상위 디렉토리

        if html_path and Path(html_path).exists():
            arcname = f"{prefix}/{job_id}_{base_name}.html"
            zf.write(html_path, arcname)

        if markdown_path and Path(markdown_path).exists():
            arcname = f"{prefix}/{job_id}_{base_name}.md"
            zf.write(markdown_path, arcname)

        if csv_path and Path(csv_path).exists():
            arcname = f"{prefix}/{job_id}_{base_name}_tables.csv"
            zf.write(csv_path, arcname)

        if pkl_path and Path(pkl_path).exists():
            arcname = f"{prefix}/{job_id}_{base_name}.pkl"
            zf.write(pkl_path, arcname)

        if image_paths:
            for img_path in image_paths:
                if Path(img_path).exists():
                    img_name = Path(img_path).name
                    arcname = f"{prefix}/images/{img_name}"
                    zf.write(img_path, arcname)

    zip_filename = f"result/{zip_name}"
    logger.info(f"[{job_id}] ZIP 생성 완료: {zip_filename}")

    return zip_filename
