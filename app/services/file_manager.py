"""파일 업로드/다운로드 관리"""

from pathlib import Path
from typing import Optional

from app.config import settings


def get_upload_dir(job_id: str) -> Path:
    """작업별 업로드 디렉토리 반환 (생성 포함)"""
    path = Path(settings.uploads_volume) / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_work_dir(job_id: str) -> Path:
    """작업별 중간 파일 디렉토리 반환 (생성 포함)"""
    path = Path(settings.data_volume) / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_result_dir() -> Path:
    """결과 저장 디렉토리 반환"""
    path = Path(settings.result_volume)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_zip_path(job_id: str, zip_filename: str) -> Optional[Path]:
    """ZIP 파일 경로 반환"""
    # zip_filename은 "result/{job_id}_{timestamp}.zip" 형식
    # result/ 프리픽스 제거 후 실제 경로 구성
    name = zip_filename.replace("result/", "", 1) if zip_filename.startswith("result/") else zip_filename
    path = get_result_dir() / name
    if path.exists():
        return path
    return None


async def save_upload(job_id: str, filename: str, content: bytes) -> Path:
    """업로드된 파일을 작업 디렉토리에 저장"""
    upload_dir = get_upload_dir(job_id)
    file_path = upload_dir / filename
    file_path.write_bytes(content)
    return file_path
