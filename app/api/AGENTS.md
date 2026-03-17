<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# api

## Purpose
FastAPI 라우터 모듈. 5개 REST API 엔드포인트를 정의하고, 파라미터 변환(string→typed) 및 백그라운드 파이프라인 실행을 담당한다.

## Key Files

| File | Description |
|------|-------------|
| `routes.py` | 5개 엔드포인트: `GET /health`, `POST /parse`, `GET /status/{job_id}`, `GET /download/{job_id}`, `GET /jobs`. 백그라운드 태스크 `_run_pipeline()` 포함 |
| `__init__.py` | 빈 패키지 초기화 |

## For AI Agents

### Working In This Directory
- `POST /parse`에서 Form 파라미터(모두 string)를 typed로 변환 — 이 변환은 반드시 routes.py에서 수행 (파이프라인 노드에서 하지 않음)
- API 키: `X-UPSTAGE-API-KEY`, `X-OPENAI-API-KEY` 헤더 우선, 환경 변수 fallback
- `_run_pipeline()`은 `BackgroundTasks`로 실행되며 job_manager를 통해 상태 업데이트
- 결과 다운로드는 `FileResponse`로 ZIP 파일 반환

### Common Patterns
- 파라미터 변환: `include_image.lower() == "true"`, `int(batch_size)` 등
- 에러: `HTTPException`으로 400/404/413 반환
- job 생명주기: pending → processing → completed/failed

## Dependencies

### Internal
- `app.config.settings` — 설정값
- `app.services.job_manager` — 작업 CRUD
- `app.services.file_manager` — 파일 저장/조회
- `app.pipeline.runner` — 파이프라인 실행 (lazy import)
- `app.utils.zip_utils` — ZIP 패키징 (lazy import)
- `app.models.schemas` — 응답 Pydantic 모델

### External
- `fastapi` (APIRouter, BackgroundTasks, File, Form, Header, HTTPException, FileResponse)

<!-- MANUAL: -->
