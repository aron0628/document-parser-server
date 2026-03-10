# Document Parser Server

## Overview
PDF 파싱 API 서버. Upstage(레이아웃 분석) + OpenAI(이미지/테이블 엔티티 추출) 파이프라인.

## Build & Run
```bash
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 9997

# Docker
docker compose up --build

# Test
pytest
```

## Architecture
- **FastAPI** 서버 (`app/main.py`)
- **LangGraph** 파이프라인 (`app/pipeline/graph.py`) — 14 nodes, 3 phases
- **File-based job storage** — `data/jobs/{job_id}.json`
- **Background task** — `_run_pipeline` in routes.py

## Client
- Python 클라이언트: `/Users/aron/Documents/lab/document-parser-client`
- GitHub: https://github.com/aron0628/document-parser-client
- 동기/비동기 클라이언트 + CLI 제공

## Key Rules
- All external HTTP calls use `httpx.AsyncClient` (no `requests`)
- Pipeline nodes are async functions taking `PipelineState`, returning `dict`
- ZIP structure: `{job_id}/{job_id}_{base_name}.ext`
- API keys: header (`X-UPSTAGE-API-KEY`) → env var fallback
- String→typed conversion happens in `routes.py`, not in pipeline nodes

## Directory Structure
```
app/
├── api/routes.py          # 5 API endpoints
├── config.py              # Pydantic Settings
├── models/                # schemas.py, state.py
├── services/              # job_manager.py, file_manager.py
├── pipeline/
│   ├── graph.py           # LangGraph StateGraph (14 nodes)
│   ├── runner.py          # PipelineRunner protocol
│   ├── nodes/             # 14 node implementations
│   └── external/          # upstage_client.py, openai_client.py
└── utils/                 # pdf_utils.py, zip_utils.py
```
