<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# external

## Purpose
외부 AI API 클라이언트. Upstage Document Digitization API와 OpenAI Vision/Chat API를 `httpx.AsyncClient`로 호출한다.

## Key Files

| File | Description |
|------|-------------|
| `upstage_client.py` | `parse_document()` — Upstage Document Digitization API로 PDF 레이아웃 분석. 지수 백오프 재시도 (최대 3회). 타임아웃 300초 |
| `openai_client.py` | `describe_image()` — OpenAI Vision으로 이미지 설명 생성 (컨텍스트 지원). `extract_table()` — 테이블을 마크다운으로 변환 (텍스트/멀티모달/이미지 3모드) |
| `__init__.py` | 빈 패키지 초기화 |

## For AI Agents

### Working In This Directory
- 모든 HTTP 호출은 `httpx.AsyncClient` 사용 (프로젝트 규칙: `requests` 금지)
- 재시도 로직: 429(Rate Limit) + 5xx(Server Error) + `RequestError` → 지수 백오프
- `_call_openai()`는 내부 공통 함수로 `describe_image()`과 `extract_table()` 모두 사용
- API 키는 노드에서 state를 통해 전달받음 (환경 변수 직접 접근 없음)

### API Endpoints
- Upstage: `https://api.upstage.ai/v1/document-digitization` (multipart/form-data)
- OpenAI: `https://api.openai.com/v1/chat/completions` (JSON)

### Table Extraction Modes
1. **텍스트+이미지** (멀티모달): HTML 텍스트 + base64 이미지 → `_TABLE_MULTIMODAL_PROMPT`
2. **이미지만**: base64 이미지 → `_TABLE_IMAGE_ONLY_PROMPT`
3. **텍스트만**: HTML 텍스트 → `_TABLE_TEXT_PROMPT`

### Common Patterns
- `async with httpx.AsyncClient(timeout=...) as client:` — 요청당 클라이언트 생성
- 인증: `Authorization: Bearer {api_key}` 헤더
- 이미지 전송: base64 인코딩 → `data:image/png;base64,{b64}` 형태

## Dependencies

### External
- `httpx` (AsyncClient), `base64`, `pathlib`

<!-- MANUAL: -->
