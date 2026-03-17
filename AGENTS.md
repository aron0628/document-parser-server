<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# document-parser-server

## Purpose
PDF 문서를 파싱하여 레이아웃 분석(Upstage API) → 이미지/테이블 AI 보강(OpenAI Vision) → 다중 형식 내보내기(HTML, Markdown, CSV, PKL)를 수행하는 FastAPI 기반 비동기 API 서버. LangGraph 기반 16노드 3-Phase 파이프라인으로 구성되며, 벡터 임베딩 생성 후 PostgreSQL(pgvector)에 저장하는 기능을 포함한다.

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | 프로젝트 의존성 및 빌드 설정 (Python ≥3.11, FastAPI, LangGraph, httpx 등) |
| `Dockerfile` | Python 3.11-slim + uv 기반 컨테이너 이미지 빌드 |
| `docker-compose.yml` | API 서버 + PostgreSQL(pgvector) 2-서비스 구성 |
| `CLAUDE.md` | AI 에이전트용 프로젝트 규칙 및 아키텍처 요약 |
| `.env` | 환경 변수 (API 키, DB 접속 정보 등) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `app/` | 애플리케이션 소스 코드 전체 (see `app/AGENTS.md`) |
| `tests/` | pytest 기반 단위/통합 테스트 (see `tests/AGENTS.md`) |
| `data/` | 런타임 생성 — 작업 JSON 파일 저장소 (`data/jobs/{job_id}.json`) |
| `result/` | 런타임 생성 — ZIP 결과 파일 저장소 |
| `uploads/` | 런타임 생성 — 업로드된 PDF 및 중간 파일 |

## For AI Agents

### Working In This Directory
- `uv sync --dev` 후 `uv run uvicorn app.main:app --host 0.0.0.0 --port 9997`로 실행
- Docker: `docker compose up --build`
- 모든 외부 HTTP 호출은 `httpx.AsyncClient` 사용 (requests 금지)
- 커밋 메시지는 한글, AI 관련 문구(Co-Authored-By 등) 금지
- 명시적 요청 없이 커밋/푸시 금지

### Testing Requirements
- `uv run pytest` — asyncio_mode="auto" 설정
- psycopg/pgvector 미설치 환경에서도 stub으로 테스트 가능

### Architecture Overview
```
[Client] → POST /parse (PDF upload)
         → Background Task: _run_pipeline()
           → LangGraph StateGraph (16 nodes, 3 phases)
             Phase 1: split_pdf → working_queue ⟲ document_parse → post_document_parse
             Phase 2: create_elements → export_image → [parallel branches]
               Branch A: page_elements_extractor → image/table_entity_extractor → merge → reconstruct
               Branch B: export_html, export_markdown, export_table_csv (parallel, un-enriched)
             Phase 3: langchain_document → embedding → END
           → ZIP packaging → Job completed
         → GET /status/{job_id} (polling)
         → GET /download/{job_id} (ZIP download)
```

## Dependencies

### External
- `fastapi` + `uvicorn` — 비동기 웹 프레임워크
- `langgraph` (0.2.x) + `langchain-core` — 파이프라인 오케스트레이션
- `httpx` — 비동기 HTTP 클라이언트 (Upstage/OpenAI API 호출)
- `openai` (AsyncOpenAI) — Upstage 임베딩 API 호출 (OpenAI 호환)
- `psycopg` + `psycopg-pool` + `pgvector` — PostgreSQL 벡터 DB
- `PyPDF2` — PDF 분할
- `Pillow` — 이미지 처리
- `langchain-text-splitters` — 텍스트 청크 분할

<!-- MANUAL: -->
