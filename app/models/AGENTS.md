<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# models

## Purpose
API 요청/응답 Pydantic 모델과 LangGraph 파이프라인 상태(TypedDict) 정의. 서버의 데이터 스키마 전체를 관리한다.

## Key Files

| File | Description |
|------|-------------|
| `schemas.py` | API 응답 모델 5개: `HealthResponse`, `ParseResponse`, `StatusResponse`, `JobSummary`, `JobListResponse` |
| `state.py` | `PipelineState(TypedDict)` — 파이프라인 전체 상태 관리. Input(9필드), Phase1(4필드), Phase2(7필드), Phase3(5필드) |
| `__init__.py` | 빈 패키지 초기화 |

## For AI Agents

### Working In This Directory
- `PipelineState`는 `total=False` TypedDict — 모든 필드가 선택적
- 새 파이프라인 노드가 상태에 데이터를 추가하면 `state.py`에 필드 추가 필요
- 병렬 브랜치는 서로 다른 키에만 기록 (키 충돌 방지)
- `schemas.py`의 모델은 API 응답 직렬화 전용 — 내부 파이프라인과 무관

### Common Patterns
- 노드 반환: `dict` (PipelineState의 부분 업데이트)
- 상태 접근: `state.get("key", default)` 패턴 (TypedDict이지만 런타임은 dict)

## Dependencies

### External
- `pydantic` (BaseModel), `typing` (TypedDict, List, Dict, Optional)

<!-- MANUAL: -->
