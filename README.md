# Document Parser Server

PDF 문서를 파싱하여 구조화된 데이터로 변환하는 FastAPI 서버입니다.

Upstage Document Parse API로 레이아웃 분석 후, OpenAI Vision API로 이미지/테이블 엔티티를 추출하고, HTML · Markdown · CSV · LangChain Document(pkl) 4가지 형식으로 내보냅니다. 추출된 Document는 텍스트 분할 후 Upstage Embedding API로 벡터화하여 PostgreSQL(pgvector)에 저장합니다.

## 아키텍처

```
Phase 1: Document Parse (50%)
  split_pdf → working_queue ⟲ document_parse → post_document_parse
  PDF를 batch_size 단위로 분할하여 Upstage API로 순차 파싱 후 결과 병합

Phase 2: Element Processing (35%)
  create_elements → export_image → [fan-out]
  ├── Branch A (entity extraction):
  │   page_elements_extractor
  │   ├── image_entity_extractor (OpenAI Vision) ──┐
  │   └── table_entity_extractor (OpenAI Chat)  ───┤
  │                                                 └── merge_entity → reconstruct_elements
  └── Branch B (quick exports, parallel):
      ├── export_html (un-enriched)
      ├── export_markdown (un-enriched)
      └── export_table_csv (un-enriched)

Phase 3: Export (15%)
  langchain_document → embedding → END
  enriched 요소를 LangChain Document으로 변환 → 텍스트 분할 → 벡터 임베딩 → DB 저장
```

### 전체 흐름

```
[Client] → POST /parse (PDF 업로드)
         → Background Task: LangGraph 파이프라인 (16 nodes, 3 phases)
         → ZIP 패키징 → Job 완료
         → GET /status/{job_id} (폴링)
         → GET /download/{job_id} (ZIP 다운로드)
```

## 빠른 시작

### Docker (권장)

```bash
cp .env.example .env
# .env 파일에 API 키 입력

docker compose up --build
```

API 서버(`localhost:9997`)와 PostgreSQL/pgvector(`localhost:5432`)가 함께 실행됩니다.

### 로컬 실행

```bash
uv sync --dev
cp .env.example .env

uv run uvicorn app.main:app --host 0.0.0.0 --port 9997
```

> DB 없이도 서버는 정상 기동됩니다. 임베딩 기능만 비활성화 상태로 동작합니다.

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/parse` | PDF 업로드 및 파싱 요청 |
| GET | `/status/{job_id}` | 작업 상태 조회 (phase, progress 포함) |
| GET | `/download/{job_id}` | 결과 ZIP 다운로드 |
| GET | `/jobs` | 전체 작업 목록 조회 |

### 파싱 요청 예시

```python
import httpx

with open("document.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:9997/parse",
        files={"file": ("document.pdf", f, "application/pdf")},
        data={
            "language": "Korean",
            "include_image": "true",
            "batch_size": "30",
        },
        headers={
            "X-UPSTAGE-API-KEY": "your_key",
            "X-OPENAI-API-KEY": "your_key",
        },
    )
job_id = response.json()["job_id"]
```

### 파싱 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `file` | File | (필수) | PDF 파일 |
| `language` | str | `"Korean"` | 처리 언어 |
| `include_image` | str | `"true"` | 이미지 내보내기 여부 |
| `batch_size` | str | `"30"` | 배치당 페이지 수 |
| `test_page` | str | `None` | 처리할 최대 페이지 수 (테스트용) |
| `embedding_model` | str | `"embedding-passage"` | Upstage 임베딩 모델명 |
| `chunk_size` | str | `"1000"` | 텍스트 분할 크기 |
| `chunk_overlap` | str | `"200"` | 텍스트 분할 오버랩 |

### 결과 ZIP 구조

```
{job_id}/
├── {job_id}_{filename}.html      # HTML 문서
├── {job_id}_{filename}.md        # Markdown 문서
├── {job_id}_{filename}_tables.csv # 테이블 CSV
├── {job_id}_{filename}.pkl       # LangChain Document (enriched)
└── images/                       # 추출된 이미지/테이블 PNG
    ├── {job_id}_page_0_img_0.png
    └── {job_id}_page_0_table_1.png
```

## 클라이언트

Python 클라이언트: [document-parser-client](https://github.com/aron0628/document-parser-client)

```python
from document_parser_client import DocumentParserClient

client = DocumentParserClient(api_url="http://localhost:9997")
job_id = client.parse_pdf("document.pdf", language="Korean")
result = client.wait_for_job_completion(job_id)
client.download_result(job_id, save_dir="./output", extract=True)
```

비동기 클라이언트도 지원:

```python
from document_parser_client import AsyncDocumentParserClient

async with AsyncDocumentParserClient(api_url="http://localhost:9997") as client:
    job_id = await client.parse_pdf("document.pdf")
    await client.wait_for_job_completion(job_id)
    await client.download_result(job_id)
```

CLI 사용:

```bash
pip install document-parser-client
document-parser parse document.pdf --wait --download
```

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `UPSTAGE_API_KEY` | O | - | Upstage API 키 (레이아웃 분석 + 임베딩) |
| `OPENAI_API_KEY` | O | - | OpenAI API 키 (이미지/테이블 엔티티 추출) |
| `PORT` | X | `9997` | 서버 포트 |
| `DB_HOST` | X | `localhost` | PostgreSQL 호스트 |
| `DB_PORT` | X | `5432` | PostgreSQL 포트 |
| `DB_NAME` | X | `document_parser` | 데이터베이스명 |
| `DB_USER` | X | `parser` | DB 사용자 |
| `DB_PASSWORD` | X | `parser` | DB 비밀번호 |

## 프로젝트 구조

```
app/
├── main.py                # FastAPI 앱 (lifespan, CORS, 라우터)
├── config.py              # Pydantic Settings
├── db.py                  # PostgreSQL 커넥션 풀 + pgvector
├── logging_config.py      # 중앙 로깅 설정
├── api/routes.py          # 5 API endpoints
├── models/
│   ├── schemas.py         # API 응답 Pydantic 모델
│   └── state.py           # PipelineState TypedDict
├── services/
│   ├── job_manager.py     # 작업 CRUD (JSON 파일 기반)
│   └── file_manager.py    # 파일/디렉토리 관리
├── pipeline/
│   ├── graph.py           # LangGraph StateGraph (16 nodes)
│   ├── runner.py          # PipelineRunner 프로토콜
│   ├── logging.py         # 진행률 추적 + 콜백 계측
│   ├── nodes/             # 16 node implementations
│   └── external/          # upstage_client.py, openai_client.py
└── utils/
    ├── pdf_utils.py       # PDF 분할 (PyPDF2)
    └── zip_utils.py       # 결과 ZIP 패키징
```

## 개발

```bash
# 테스트
uv run pytest

# 린트
ruff check app/
black --check app/
```

## 라이선스

MIT
