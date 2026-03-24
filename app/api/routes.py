"""Document Parser API 엔드포인트 (7개)"""

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
    DeleteJobResponse,
    HealthResponse,
    JobListResponse,
    JobSummary,
    ParseResponse,
    RaptorRetryResponse,
    ResumeResponse,
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


def _finalize_job(job_id: str, result_state: dict) -> None:
    """파이프라인 완료 후 ZIP 패키징 + job 상태 업데이트 (공통)"""
    from app.utils.zip_utils import create_result_zip

    job = job_manager.get_job(job_id)
    original_filename = job["filename"] if job else "unknown.pdf"

    zip_filename = create_result_zip(
        job_id=job_id,
        original_filename=original_filename,
        html_path=result_state.get("html_path"),
        markdown_path=result_state.get("markdown_path"),
        csv_path=result_state.get("csv_path"),
        pkl_path=result_state.get("pkl_path"),
        image_paths=result_state.get("image_paths"),
    )

    # RAPTOR 상태 판단
    raptor_level_counts = result_state.get("raptor_level_counts")
    enable_raptor = result_state.get("enable_raptor", False)

    if not enable_raptor:
        raptor_status = "skipped"
    elif raptor_level_counts:  # 비어있지 않은 dict = 성공
        raptor_status = "success"
    else:
        raptor_status = "failed"

    job_manager.update_job(
        job_id,
        status="completed",
        completed_at=time.time(),
        zip_filename=zip_filename,
        raptor_status=raptor_status,
    )


async def _handle_pipeline_failure(job_id: str, error: Exception) -> None:
    """파이프라인 실패 시 job 상태 업데이트 (공통)"""
    logger.error(f"Pipeline failed for job {job_id}: {error}\n{traceback.format_exc()}")
    try:
        job_manager.update_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            error=str(error),
        )
    except Exception as update_err:
        logger.error(f"Failed to update job status for {job_id}: {update_err}")
        # 재시도: 잠시 대기 후 FD 회수된 뒤 다시 시도
        await asyncio.sleep(2)
        try:
            job_manager.update_job(
                job_id,
                status="failed",
                completed_at=time.time(),
                error=str(error),
            )
        except Exception:
            logger.critical(
                f"Job {job_id} stuck in processing state - "
                f"manual intervention required"
            )


async def _run_pipeline(job_id: str, pdf_path: str, params: dict) -> None:
    """백그라운드에서 파이프라인 실행 + ZIP 패키징"""
    try:
        job_manager.update_job(job_id, status="processing")

        from app.pipeline.runner import run_pipeline

        result_state = await run_pipeline(job_id, pdf_path, params)
        _finalize_job(job_id, result_state)
    except Exception as e:
        await _handle_pipeline_failure(job_id, e)


async def _resume_pipeline(job_id: str, api_keys: dict) -> None:
    """checkpoint에서 파이프라인 재개"""
    try:
        job_manager.update_job(job_id, status="processing", error=None)

        from app.pipeline.runner import get_runner
        from app.pipeline.logging import LangGraphCallbackTracker, PipelineTracker

        runner = get_runner()

        config = {
            "configurable": {
                "thread_id": job_id,
                "upstage_api_key": api_keys["upstage_api_key"],
                "openai_api_key": api_keys["openai_api_key"],
            },
        }

        # checkpoint에서 state 복원하여 recursion_limit 계산
        saved_state = await runner._graph.aget_state(config)

        if saved_state and saved_state.values:
            batch_count = len(saved_state.values.get("pdf_chunks", []))
            recursion_limit = max(100, batch_count * 3 + 50)
        else:
            recursion_limit = 100

        tracker = PipelineTracker(job_id, on_phase_change=runner._make_on_phase_change(job_id))
        callback = LangGraphCallbackTracker(tracker, state=saved_state.values if saved_state else {})

        config["callbacks"] = [callback]
        config["recursion_limit"] = recursion_limit

        # checkpoint에서 재개 (None 전달 = 마지막 checkpoint부터)
        result_state = await runner._graph.ainvoke(None, config)
        _finalize_job(job_id, result_state)
    except Exception as e:
        await _handle_pipeline_failure(job_id, e)


@router.post("/parse", status_code=200)
async def parse_pdf(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("Korean"),
    include_image: str = Form("true"),
    batch_size: str = Form("30"),
    test_page: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    chunk_overlap: Optional[str] = Form(None),
    enable_raptor: Optional[str] = Form(None),
    enable_keyword_extraction: bool = Form(default=True),
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
        "embedding_model": embedding_model or settings.default_embedding_model,
        "chunk_size": int(chunk_size) if chunk_size is not None else settings.chunk_size,
        "chunk_overlap": int(chunk_overlap) if chunk_overlap is not None else settings.chunk_overlap,
        "enable_raptor": enable_raptor.lower() == "true" if enable_raptor is not None else settings.enable_raptor,
        "enable_keyword_extraction": enable_keyword_extraction,
        "upstage_api_key": upstage_key,
        "openai_api_key": openai_key,
    }

    # API 키를 params에서 분리 (job JSON에 평문 저장 방지)
    api_keys = {
        "upstage_api_key": params.pop("upstage_api_key"),
        "openai_api_key": params.pop("openai_api_key"),
    }

    # 작업 생성
    job = job_manager.create_job(filename=file.filename or "unknown.pdf", params=params)
    job_id = job["job_id"]

    # 파일 저장
    pdf_path = await file_manager.save_upload(job_id, file.filename or "upload.pdf", content)

    # 백그라운드 파이프라인 실행
    background_tasks.add_task(_run_pipeline, job_id, str(pdf_path), {**params, **api_keys})

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
        raptor_status=job.get("raptor_status"),
        raptor_error=job.get("raptor_error"),
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


@router.post("/resume/{job_id}", status_code=200)
async def resume_pipeline(
    background_tasks: BackgroundTasks,
    job_id: str,
    x_upstage_api_key: Optional[str] = Header(None, alias="X-UPSTAGE-API-KEY"),
    x_openai_api_key: Optional[str] = Header(None, alias="X-OPENAI-API-KEY"),
) -> ResumeResponse:
    """실패한 작업을 마지막 checkpoint에서 재개"""
    from app.pipeline.checkpointer import get_checkpointer

    # 1. checkpointer 활성화 확인
    checkpointer = get_checkpointer()
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="Checkpointer가 활성화되지 않았습니다.")

    # 2. job 상태 확인
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"작업 ID '{job_id}'를 찾을 수 없습니다.")
    if job["status"] != "failed":
        raise HTTPException(status_code=400, detail=f"재개는 실패한 작업만 가능합니다. 현재 상태: {job['status']}")

    # 3. API 키 확보
    upstage_key = x_upstage_api_key or settings.upstage_api_key
    openai_key = x_openai_api_key or settings.openai_api_key
    if not upstage_key or not openai_key:
        raise HTTPException(status_code=400, detail="UPSTAGE API 키와 OpenAI API 키가 필요합니다.")

    # 4. checkpoint 존재 확인
    from app.pipeline.runner import get_runner

    runner = get_runner()
    config = {"configurable": {"thread_id": job_id}}
    try:
        saved_state = await runner._graph.aget_state(config)
        if not saved_state or not saved_state.values:
            raise HTTPException(
                status_code=404,
                detail="저장된 checkpoint가 없습니다. /parse로 새 작업을 요청하세요.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Checkpoint 조회 실패: {e}")

    # 5. 백그라운드 재개
    api_keys = {"upstage_api_key": upstage_key, "openai_api_key": openai_key}
    background_tasks.add_task(_resume_pipeline, job_id, api_keys)

    return ResumeResponse(
        job_id=job_id,
        status="processing",
        message="파이프라인 재개가 시작되었습니다.",
    )


async def _run_raptor_retry(job_id: str, api_keys: dict) -> None:
    """RAPTOR 노드만 재실행하는 wrapper"""
    from app.pipeline.nodes.raptor import raptor_node
    from langchain_core.runnables import RunnableConfig

    try:
        job_manager.update_job(job_id, raptor_status="processing", raptor_error=None)

        job = job_manager.get_job(job_id)
        params = job.get("params", {})

        # raptor_node에 필요한 최소 state
        state = {
            "job_id": job_id,
            "enable_raptor": True,
            "embedding_model": params.get("embedding_model", settings.default_embedding_model),
        }

        config = RunnableConfig(configurable={
            "thread_id": job_id,
            "upstage_api_key": api_keys["upstage_api_key"],
            "openai_api_key": api_keys["openai_api_key"],
        })

        result = await raptor_node(state, config)

        raptor_level_counts = result.get("raptor_level_counts", {})
        if raptor_level_counts:
            job_manager.update_job(job_id, raptor_status="success", raptor_error=None)
        else:
            job_manager.update_job(job_id, raptor_status="failed", raptor_error="재시도 후에도 결과 없음")

    except Exception as e:
        logger.error(f"[{job_id}] RAPTOR retry 실패: {e}")
        job_manager.update_job(job_id, raptor_status="failed", raptor_error=str(e))


@router.post("/retry-raptor/{job_id}", status_code=200)
async def retry_raptor(
    background_tasks: BackgroundTasks,
    job_id: str,
    x_upstage_api_key: Optional[str] = Header(None, alias="X-UPSTAGE-API-KEY"),
    x_openai_api_key: Optional[str] = Header(None, alias="X-OPENAI-API-KEY"),
) -> RaptorRetryResponse:
    """completed job에서 RAPTOR만 재실행"""
    # 1. job 존재 확인
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"작업 ID '{job_id}'를 찾을 수 없습니다.")

    # 2. job status 확인
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"RAPTOR 재실행은 완료된 작업만 가능합니다. 현재 상태: {job['status']}")

    # 3. raptor_status 확인
    current_raptor_status = job.get("raptor_status")
    if current_raptor_status == "processing":
        raise HTTPException(status_code=409, detail="RAPTOR 재실행이 이미 진행 중입니다.")
    if current_raptor_status == "success":
        raise HTTPException(status_code=400, detail="RAPTOR가 이미 성공적으로 완료되었습니다.")

    # 4. API 키
    upstage_key = x_upstage_api_key or settings.upstage_api_key
    openai_key = x_openai_api_key or settings.openai_api_key
    if not upstage_key or not openai_key:
        raise HTTPException(status_code=400, detail="UPSTAGE API 키와 OpenAI API 키가 필요합니다.")

    # 5. 백그라운드 실행
    api_keys = {"upstage_api_key": upstage_key, "openai_api_key": openai_key}
    background_tasks.add_task(_run_raptor_retry, job_id, api_keys)

    return RaptorRetryResponse(
        job_id=job_id,
        raptor_status="processing",
        message="RAPTOR 재실행이 시작되었습니다.",
    )


@router.delete("/jobs/{job_id}", status_code=200)
async def delete_job(job_id: str) -> DeleteJobResponse:
    """특정 작업의 모든 데이터 삭제

    - document_embeddings, raptor_summaries DB 레코드 삭제
    - LangGraph checkpoint 삭제
    - 결과 ZIP 파일 삭제
    - 업로드/작업 디렉토리 삭제
    - 작업 JSON 파일 삭제

    job 파일이 없어도 DB 정리는 시도한다.
    """
    import shutil

    from app.db import get_pool
    from app.pipeline.checkpointer import cleanup_job_checkpoints

    embeddings_deleted = 0
    raptor_summaries_deleted = 0
    checkpoints_deleted = False
    result_file_deleted = False
    upload_dir_deleted = False
    work_dir_deleted = False

    # 1. DB: document_embeddings 삭제
    try:
        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM document_embeddings WHERE job_id = %s",
                    (job_id,),
                )
                embeddings_deleted = cur.rowcount if cur.rowcount >= 0 else 0
            await conn.commit()
    except Exception as e:
        logger.warning(f"[{job_id}] document_embeddings 삭제 실패 (계속 진행): {e}")

    # 2. DB: raptor_summaries 삭제
    try:
        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM raptor_summaries WHERE job_id = %s",
                    (job_id,),
                )
                raptor_summaries_deleted = cur.rowcount if cur.rowcount >= 0 else 0
            await conn.commit()
    except Exception as e:
        logger.warning(f"[{job_id}] raptor_summaries 삭제 실패 (계속 진행): {e}")

    # 3. DB: document_keywords 삭제
    try:
        pool = get_pool()
        if pool:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM document_keywords WHERE job_id = %s",
                        (job_id,),
                    )
                await conn.commit()
    except Exception as e:
        logger.warning(f"[{job_id}] document_keywords 삭제 실패 (계속 진행): {e}")

    # 4. Checkpoint 삭제
    try:
        await cleanup_job_checkpoints(job_id)
        checkpoints_deleted = True
    except Exception as e:
        logger.warning(f"[{job_id}] checkpoint 삭제 실패 (계속 진행): {e}")

    # 5. 결과 ZIP 파일 삭제
    job = job_manager.get_job(job_id)
    if job:
        zip_filename = job.get("zip_filename")
        if zip_filename:
            zip_path = file_manager.get_zip_path(job_id, zip_filename)
            if zip_path is not None:
                try:
                    zip_path.unlink()
                    result_file_deleted = True
                except Exception as e:
                    logger.warning(f"[{job_id}] ZIP 파일 삭제 실패 (계속 진행): {e}")

    # 6. 업로드 디렉토리 삭제 ({uploads_volume}/{job_id}/)
    upload_dir = file_manager.get_upload_dir(job_id)
    if upload_dir.exists():
        try:
            shutil.rmtree(upload_dir)
            upload_dir_deleted = True
        except Exception as e:
            logger.warning(f"[{job_id}] 업로드 디렉토리 삭제 실패 (계속 진행): {e}")

    # 7. 작업 중간 파일 디렉토리 삭제 ({data_volume}/{job_id}/)
    work_dir = file_manager.get_work_dir(job_id)
    if work_dir.exists():
        try:
            shutil.rmtree(work_dir)
            work_dir_deleted = True
        except Exception as e:
            logger.warning(f"[{job_id}] 작업 디렉토리 삭제 실패 (계속 진행): {e}")

    # 8. 작업 JSON 파일 삭제
    job_file_deleted = job_manager.delete_job(job_id)

    logger.info(
        f"[{job_id}] 작업 삭제 완료: "
        f"job_file={job_file_deleted}, "
        f"embeddings={embeddings_deleted}, "
        f"raptor_summaries={raptor_summaries_deleted}, "
        f"checkpoints={checkpoints_deleted}, "
        f"result_zip={result_file_deleted}, "
        f"upload_dir={upload_dir_deleted}, "
        f"work_dir={work_dir_deleted}"
    )

    return DeleteJobResponse(
        job_id=job_id,
        job_file_deleted=job_file_deleted,
        embeddings_deleted=embeddings_deleted,
        raptor_summaries_deleted=raptor_summaries_deleted,
        checkpoints_deleted=checkpoints_deleted,
        result_file_deleted=result_file_deleted,
        upload_dir_deleted=upload_dir_deleted,
        work_dir_deleted=work_dir_deleted,
    )
