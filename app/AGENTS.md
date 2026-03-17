<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# app

## Purpose
FastAPI 애플리케이션 패키지. API 엔드포인트, 설정, 데이터 모델, 서비스 계층, LangGraph 파이프라인을 포함하는 메인 소스 디렉토리.

## Key Files

| File | Description |
|------|-------------|
| `main.py` | FastAPI 앱 인스턴스 생성, lifespan(디렉토리 생성 + DB 초기화), CORS 미들웨어, 라우터 등록 |
| `config.py` | `pydantic_settings.BaseSettings` 기반 설정 (API 키, 포트, 볼륨 경로, 임베딩, DB) |
| `db.py` | PostgreSQL `AsyncConnectionPool` 관리 + pgvector 타입 등록. DB 실패 시 서버는 정상 기동(임베딩만 비활성화) |
| `logging_config.py` | `app` 네임스페이스 로거 초기화 (`setup_logging()`) |
| `__init__.py` | 빈 패키지 초기화 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `api/` | FastAPI 라우터 및 5개 API 엔드포인트 (see `api/AGENTS.md`) |
| `models/` | Pydantic 스키마 + LangGraph 파이프라인 상태 TypedDict (see `models/AGENTS.md`) |
| `services/` | 작업 관리(JSON 파일) 및 파일 I/O 서비스 (see `services/AGENTS.md`) |
| `pipeline/` | LangGraph 파이프라인 그래프, 노드, 외부 API 클라이언트 (see `pipeline/AGENTS.md`) |
| `utils/` | PDF 분할, ZIP 패키징 유틸리티 (see `utils/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- `main.py`의 lifespan에서 디렉토리 생성 + DB 초기화가 수행됨
- `config.py` 설정값은 `.env` 파일 또는 환경 변수로 오버라이드 가능
- DB 연결 실패는 non-fatal — 서버 기동은 유지되고 임베딩만 비활성화
- 새 모듈 추가 시 적절한 하위 패키지에 배치할 것

### Common Patterns
- 모든 비동기 함수는 `async def` + `await` 패턴
- 설정 접근: `from app.config import settings`
- 로깅: `logger = logging.getLogger(__name__)`
- DB 접근: `from app.db import get_pool` → `async with pool.connection() as conn:`

## Dependencies

### Internal
- 모든 하위 패키지가 `app.config.settings`에 의존
- `app.db`는 `app.config`에 의존
- `app.api.routes`는 `app.services`, `app.pipeline`, `app.models`에 의존

### External
- `fastapi`, `pydantic-settings`, `psycopg-pool`, `pgvector`

<!-- MANUAL: -->
