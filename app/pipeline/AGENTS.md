<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# pipeline

## Purpose
LangGraph 기반 PDF 처리 파이프라인. 16개 노드를 3-Phase StateGraph로 구성하며, 외부 API 클라이언트(Upstage, OpenAI)와 구조화된 로깅/추적 시스템을 포함한다.

## Key Files

| File | Description |
|------|-------------|
| `graph.py` | `build_graph()` — LangGraph StateGraph 정의. 16노드, conditional edge(working_queue), parallel fan-out/fan-in |
| `runner.py` | `PipelineRunner` 프로토콜 + `LangGraphPipelineRunner` 구현. 싱글턴 인스턴스. `run_pipeline()` 편의 함수 |
| `logging.py` | `PipelineTracker` (진행률/타이밍 추적) + `LangGraphCallbackTracker` (콜백 기반 노드 계측) |
| `__init__.py` | 빈 패키지 초기화 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `nodes/` | 16개 파이프라인 노드 구현 (see `nodes/AGENTS.md`) |
| `external/` | Upstage, OpenAI 외부 API 클라이언트 (see `external/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- 노드 함수는 순수 `async (PipelineState) -> dict` 함수 — LangGraph에 의존하지 않음
- `graph.py`의 노드/엣지 구조 변경 시 `logging.py`의 `PIPELINE_NODES` 리스트도 함께 업데이트
- `runner.py`의 `recursion_limit`은 배치 수에 비례하여 동적 계산
- Phase 가중치: Phase1=50%, Phase2=35%, Phase3=15%

### Graph Structure
```
Phase 1 (Document Parse):
  split_pdf → working_queue ⟲ document_parse → post_document_parse
  (working_queue는 conditional edge로 루프 제어)

Phase 2 (Element Processing):
  create_elements → export_image → [fan-out]
    Branch A: page_elements_extractor → [parallel]
      image_entity_extractor ─┐
      table_entity_extractor ─┤→ merge_entity → reconstruct_elements
    Branch B (parallel, un-enriched):
      export_html → END
      export_markdown → END
      export_table_csv → END

Phase 3 (Export):
  langchain_document → embedding → END
```

### Common Patterns
- 콜백 주입: `graph.ainvoke(state, config={"callbacks": [tracker]})`
- Phase 전환 감지: `LangGraphCallbackTracker`가 `graph:step:N` 태그로 노드 식별
- 싱글턴 그래프: `get_runner()` → 최초 1회만 `build_graph()` 호출

## Dependencies

### Internal
- `app.models.state.PipelineState` — 상태 타입
- `app.pipeline.nodes.*` — 16개 노드 함수
- `app.services.job_manager` — Phase 전환 시 작업 상태 갱신

### External
- `langgraph` (StateGraph, END), `langchain-core` (BaseCallbackHandler)

<!-- MANUAL: -->
