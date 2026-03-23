"""API 요청/응답 Pydantic 모델"""

from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ParseResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: float
    completed_at: Optional[float] = None
    zip_filename: Optional[str] = None
    error: Optional[str] = None
    current_phase: Optional[int] = None
    current_node: Optional[str] = None
    progress: Optional[float] = None
    raptor_status: Optional[str] = None
    raptor_error: Optional[str] = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: float
    completed_at: Optional[float] = None


class JobListResponse(BaseModel):
    jobs: List[JobSummary]


class ResumeResponse(BaseModel):
    """파이프라인 재개 응답"""
    job_id: str
    status: str
    message: str


class RaptorRetryResponse(BaseModel):
    """RAPTOR 재실행 응답"""
    job_id: str
    raptor_status: str
    message: str


class DeleteJobResponse(BaseModel):
    """작업 삭제 응답"""
    job_id: str
    job_file_deleted: bool
    embeddings_deleted: int
    raptor_summaries_deleted: int
    checkpoints_deleted: bool
    result_file_deleted: bool
    upload_dir_deleted: bool
    work_dir_deleted: bool
