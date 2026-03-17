<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# tests

## Purpose
pytest 기반 단위 테스트 및 통합 테스트. 파이프라인 노드의 순수 로직 검증과 그래프 구조 검증에 중점.

## Key Files

| File | Description |
|------|-------------|
| `test_create_elements.py` | `create_elements_node` 단위 테스트 — base64 보존, strip, 타입 변환 검증 |
| `test_embedding.py` | `embedding_node` 단위/통합 테스트 — 텍스트 분할, 배치 임베딩, DB 저장, 에러 핸들링 |
| `test_export_image.py` | `export_image_node` 테스트 |
| `test_image_context.py` | 이미지 컨텍스트 생성 관련 테스트 |
| `test_table_multimodal.py` | 멀티모달 테이블 추출 테스트 |
| `__init__.py` | 빈 패키지 초기화 |

## For AI Agents

### Working In This Directory
- `uv run pytest` 또는 `uv run pytest tests/test_specific.py`로 실행
- `asyncio_mode = "auto"` 설정 — `@pytest.mark.asyncio` 없이도 async 테스트 실행 가능
- psycopg/pgvector가 미설치된 환경에서는 `test_embedding.py` 상단의 stub 등록 패턴 참조
- 외부 API 호출은 `unittest.mock.patch`로 mock 처리

### Testing Requirements
- 새 노드 추가 시 해당 노드의 순수 로직 테스트 필수
- mock/stub 패턴: `patch("app.pipeline.nodes.xxx.external_func")` 형태
- `tmp_path` fixture 활용하여 임시 파일 생성/정리

### Common Patterns
- 노드 함수 직접 호출: `result = await some_node(state_dict)`
- 내부 함수 직접 테스트: `from app.pipeline.nodes.xxx import _internal_func`
- DB mock: `AsyncMock` + `__aenter__`/`__aexit__` 패턴

## Dependencies

### Internal
- `app.pipeline.nodes.*` — 테스트 대상 노드 함수
- `app.pipeline.graph` — 그래프 구조 통합 테스트

### External
- `pytest`, `pytest-asyncio`, `unittest.mock`, `langchain-core`

<!-- MANUAL: -->
