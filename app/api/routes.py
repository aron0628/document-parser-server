"""Document Parser API 엔드포인트 (5개)"""

import asyncio
import logging
import time
import traceback
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import (
    HealthResponse,
    JobListResponse,
    JobSummary,
    ParseResponse,
    StatusResponse,
)
from app.services import file_manager, job_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> HealthResponse:
    """API 서버 건강 상태 확인"""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
    )


async def _run_pipeline(job_id: str, pdf_path: str, params: dict) -> None:
    """백그라운드에서 파이프라인 실행 + ZIP 패키징"""
    try:
        job_manager.update_job(job_id, status="processing")

        from app.pipeline.runner import run_pipeline
        from app.utils.zip_utils import create_result_zip

        result_state = await run_pipeline(job_id, pdf_path, params)

        # 원본 파일명 추출
        job = job_manager.get_job(job_id)
        original_filename = job["filename"] if job else "unknown.pdf"

        # ZIP 패키징
        zip_filename = create_result_zip(
            job_id=job_id,
            original_filename=original_filename,
            html_path=result_state.get("html_path"),
            markdown_path=result_state.get("markdown_path"),
            csv_path=result_state.get("csv_path"),
            pkl_path=result_state.get("pkl_path"),
            image_paths=result_state.get("image_paths"),
        )

        job_manager.update_job(
            job_id,
            status="completed",
            completed_at=time.time(),
            zip_filename=zip_filename,
        )
    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {e}\n{traceback.format_exc()}")
        job_manager.update_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            error=str(e),
        )


@router.post("/parse", status_code=200)
async def parse_pdf(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("Korean"),
    include_image: str = Form("true"),
    batch_size: str = Form("30"),
    test_page: Optional[str] = Form(None),
    x_upstage_api_key: Optional[str] = Header(None, alias="X-UPSTAGE-API-KEY"),
    x_openai_api_key: Optional[str] = Header(None, alias="X-OPENAI-API-KEY"),
) -> ParseResponse:
    """PDF 파일 업로드 및 파싱 작업 요청"""
    # API 키 확인 (헤더 우선, 환경 변수 fallback)
    upstage_key = x_upstage_api_key or settings.upstage_api_key
    openai_key = x_openai_api_key or settings.openai_api_key

    if not upstage_key or not openai_key:
        raise HTTPException(status_code=400, detail="UPSTAGE API 키와 OpenAI API 키가 필요합니다.")

    # 파일 크기 확인
    content = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"파일 크기가 {settings.max_upload_size_mb}MB를 초과합니다.")

    # 파라미터 변환 (string → typed, routes.py가 담당)
    params = {
        "language": language,
        "include_image": include_image.lower() == "true",
        "batch_size": int(batch_size),
        "test_page": int(test_page) if test_page is not None else None,
        "upstage_api_key": upstage_key,
        "openai_api_key": openai_key,
    }

    # 작업 생성
    job = job_manager.create_job(filename=file.filename or "unknown.pdf", params=params)
    job_id = job["job_id"]

    # 파일 저장
    pdf_path = await file_manager.save_upload(job_id, file.filename or "upload.pdf", content)

    # 백그라운드 파이프라인 실행
    background_tasks.add_task(_run_pipeline, job_id, str(pdf_path), params)

    return ParseResponse(
        job_id=job_id,
        status="pending",
        message="PDF 파싱 작업이 시작되었습니다.",
    )


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> StatusResponse:
    """작업 상태 확인"""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"작업 ID '{job_id}'를 찾을 수 없습니다.")

    return StatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        filename=job["filename"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        zip_filename=job.get("zip_filename"),
        error=job.get("error"),
        current_phase=job.get("current_phase"),
        current_node=job.get("current_node"),
        progress=job.get("progress"),
    )


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """작업 결과 ZIP 다운로드"""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"작업 ID '{job_id}'를 찾을 수 없습니다.")

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"작업이 완료되지 않았습니다. 현재 상태: {job['status']}")

    zip_filename = job.get("zip_filename")
    if not zip_filename:
        raise HTTPException(status_code=404, detail="ZIP 파일 경로가 없습니다.")

    zip_path = file_manager.get_zip_path(job_id, zip_filename)
    if zip_path is None:
        raise HTTPException(status_code=404, detail="ZIP 파일을 찾을 수 없습니다.")

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=zip_path.name,
    )


@router.get("/jobs")
async def list_all_jobs() -> JobListResponse:
    """모든 작업 목록 조회"""
    jobs = job_manager.list_jobs()
    return JobListResponse(
        jobs=[
            JobSummary(
                job_id=j["job_id"],
                status=j["status"],
                filename=j["filename"],
                created_at=j["created_at"],
                completed_at=j.get("completed_at"),
            )
            for j in jobs
        ]
    )
