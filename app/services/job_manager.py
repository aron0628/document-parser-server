"""작업 생명주기 관리 - 파일 기반 JSON 저장소"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings


def _jobs_dir() -> Path:
    """작업 JSON 파일 저장 디렉토리"""
    path = Path(settings.data_volume) / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_job(job: Dict[str, Any]) -> None:
    path = _job_path(job["job_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def create_job(
    filename: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """새 작업 생성"""
    job = {
        "job_id": str(uuid.uuid4()),
        "status": "pending",
        "filename": filename,
        "created_at": time.time(),
        "completed_at": None,
        "zip_filename": None,
        "error": None,
        "params": params,
    }
    _write_job(job)
    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """작업 조회"""
    return _read_job(job_id)


def update_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    """작업 상태 업데이트"""
    job = _read_job(job_id)
    if job is None:
        return None
    job.update(updates)
    _write_job(job)
    return job


def delete_job(job_id: str) -> bool:
    """작업 JSON 파일 삭제. 삭제 성공 시 True, 파일 없으면 False 반환"""
    path = _job_path(job_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def list_jobs() -> List[Dict[str, Any]]:
    """모든 작업 목록 조회"""
    jobs = []
    jobs_dir = _jobs_dir()
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                jobs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return jobs
