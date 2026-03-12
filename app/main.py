"""Document Parser API Server - FastAPI 메인 모듈"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 lifespan 이벤트"""
    setup_logging()
    # Startup: 필요한 디렉토리 생성
    for dir_path in [
        Path(settings.data_volume) / "jobs",
        Path(settings.result_volume),
        Path(settings.uploads_volume),
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="Document Parser API",
    description="PDF 문서 파싱 API 서버",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)
