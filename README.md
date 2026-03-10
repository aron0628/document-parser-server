# Document Parser Server

PDF 문서를 파싱하여 구조화된 데이터로 변환하는 FastAPI 서버입니다.

Upstage Document Parse API로 레이아웃 분석 후, OpenAI Vision API로 이미지/테이블 엔티티를 추출하고, HTML · Markdown · CSV · LangChain Document(pkl) 4가지 형식으로 내보냅니다.

## 아키텍처

```
Phase 1: Document Parse
  PDF Split → Batch Queue Loop → Upstage Parse → Merge

Phase 2: Element Processing
  Page Elements Extractor
  ├── Image Entity Extractor ──┐
  ├── Table Entity Extractor ──┤
  │                            └── Merge Entity → Reconstruct
  ├── Export HTML (un-enriched)
  ├── Export Markdown (un-enriched)
  └── Export Table CSV (un-enriched)

Phase 3: Export
  LangChain Document (enriched → pkl)
  → Final Merge
```

## 빠른 시작

### Docker (권장)

```bash
cp .env.example .env
# .env 파일에 API 키 입력

docker compose up --build
```

### 로컬 실행

```bash
pip install -e ".[dev]"
cp .env.example .env

uvicorn app.main:app --host 0.0.0.0 --port 9997
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/parse` | PDF 업로드 및 파싱 요청 |
| GET | `/status/{job_id}` | 작업 상태 조회 |
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
| `UPSTAGE_API_KEY` | O | - | Upstage API 키 |
| `OPENAI_API_KEY` | O | - | OpenAI API 키 |
| `PORT` | X | 9997 | 서버 포트 |
| `MAX_UPLOAD_SIZE_MB` | X | 100 | 최대 업로드 크기(MB) |

## 개발

```bash
# 테스트
pytest

# 린트
ruff check app/
black --check app/
```

## 라이선스

MIT
