<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-20 -->

# nodes

## Purpose
LangGraph 파이프라인의 16개 노드 구현. 각 노드는 순수 async 함수로 `PipelineState`를 받아 상태 업데이트 dict를 반환한다. 3-Phase 구조로 구분된다.

## Key Files

### Phase 1: Document Parse
| File | Description |
|------|-------------|
| `split_pdf.py` | PDF를 batch_size 단위로 분할 → `pdf_chunks` 리스트 생성 |
| `working_queue.py` | 배치 루프 컨트롤러. `check_queue()` conditional edge 함수 포함 |
| `document_parse.py` | 현재 배치의 PDF 청크를 Upstage API로 파싱 → `batch_parse_results`에 추가 |
| `post_document_parse.py` | 모든 배치 결과 병합 (페이지 오프셋 적용) → `merged_parse_result` 생성 + 청크 파일 정리 |

### Phase 2: Element Processing
| File | Description |
|------|-------------|
| `create_elements.py` | Upstage 결과에서 표준 요소 객체 리스트 생성. 카테고리 정규화(figure→image, paragraph→text). base64를 elements에 보존하고 merged에서 제거 |
| `export_image.py` | 이미지/테이블 base64 데이터를 PNG 파일로 저장. `include_image=False`이면 no-op. fallback PNG(1x1 투명) 지원 |
| `page_elements_extractor.py` | 요소를 페이지별로 그룹화하고 images/tables로 분류 → `page_elements` dict |
| `image_entity_extractor.py` | OpenAI Vision으로 이미지 설명 생성. 같은 페이지 텍스트를 컨텍스트로 제공 (최대 2000자) |
| `table_entity_extractor.py` | OpenAI로 테이블을 마크다운 테이블로 변환. 텍스트+이미지/이미지만/텍스트만 3가지 모드 |
| `merge_entity.py` | AI 분석 결과(image_entities, table_entities)를 원본 요소에 `entity` 키로 병합 |
| `reconstruct_elements.py` | 강화된 요소를 (page, position) 순서로 정렬하여 문서 구조 재구성 |

### Phase 3: Export
| File | Description |
|------|-------------|
| `export_html.py` | un-enriched 요소로 HTML 문서 생성 (페이지별 div 구조) |
| `export_markdown.py` | un-enriched 요소로 Markdown 문서 생성 (페이지별 `## Page N` 구분) |
| `export_table_csv.py` | un-enriched 테이블 요소를 CSV로 내보내기 (page, table_index, content) |
| `langchain_document.py` | enriched 요소를 LangChain `Document` 객체로 변환 후 pickle 저장. 이미지는 description, 테이블은 structured_table 사용 |
| `embedding.py` | pickle에서 Document 로드 → `RecursiveCharacterTextSplitter`로 분할 → Upstage 임베딩 → PostgreSQL 저장. **Triple try/except**: API 실패·DB 실패 모두 `embedding_count: 0` 반환 (파이프라인 중단 없음) |

| File | Description |
|------|-------------|
| `__init__.py` | 빈 패키지 초기화 |

## For AI Agents

### Working In This Directory
- 모든 노드 함수 시그니처: `async def xxx_node(state: PipelineState) -> dict`
- 반환 dict는 PipelineState의 키 부분 집합 — LangGraph가 상태에 병합
- 병렬 브랜치의 노드는 서로 다른 키에만 기록해야 함 (키 충돌 시 LangGraph 에러)
- 새 노드 추가 시: (1) 노드 파일 생성, (2) `graph.py`에 등록, (3) `logging.py`의 `PIPELINE_NODES`에 추가, (4) `state.py`에 필요한 키 추가

### Testing Requirements
- 노드의 순수 로직 함수(`_parse_elements`, `_split_documents` 등)를 별도로 단위 테스트
- async 노드 테스트: `await node_func(state_dict)` 직접 호출
- 외부 API 의존 노드: mock 필수 (`patch("app.pipeline.external.xxx")`)

### Common Patterns
- 상태 읽기: `state.get("key", default)` — KeyError 방지
- 리스트 누적: `list(state.get("key", []))` 후 append (불변성)
- 로깅: `logger.info(f"[{job_id}] ...작업 완료: N개")` 형태

### Parallel Branch Key Separation
```
export_image → [fan-out]
  Branch A (entity extraction):
    page_elements → page_elements (new key)
    image_entity_extractor → image_entities (new key)
    table_entity_extractor → table_entities (new key)
    merge_entity → merged_elements (new key)
    reconstruct_elements → reconstructed_elements (new key)
  Branch B (quick exports):
    export_html → html_path
    export_markdown → markdown_path
    export_table_csv → csv_path
```

## Dependencies

### Internal
- `app.models.state.PipelineState` — 상태 타입
- `app.pipeline.external.upstage_client` — Upstage API (document_parse)
- `app.pipeline.external.openai_client` — OpenAI API (image_entity, table_entity)
- `app.services.file_manager` — 디렉토리 경로
- `app.config.settings` — 임베딩 설정
- `app.db.get_pool` — DB 연결 (embedding)

### External
- `langchain-core` (Document), `langchain-text-splitters` (RecursiveCharacterTextSplitter)
- `openai` (AsyncOpenAI), `PyPDF2`, `Pillow`, `pickle`, `csv`

<!-- MANUAL: -->
