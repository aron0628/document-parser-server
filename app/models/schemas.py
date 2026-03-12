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


class JobSummary(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: float
    completed_at: Optional[float] = None


class JobListResponse(BaseModel):
    jobs: List[JobSummary]
