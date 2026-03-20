<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-20 -->

# services

## Purpose
작업(Job) 생명주기 관리와 파일 I/O 서비스. 파일 기반 JSON 저장소로 작업 상태를 관리하고, 업로드/작업/결과 디렉토리를 관리한다.

## Key Files

| File | Description |
|------|-------------|
| `job_manager.py` | 작업 CRUD — `create_job()`, `get_job()`, `update_job()`, `list_jobs()`. `data/jobs/{job_id}.json` 파일 기반 저장 |
| `file_manager.py` | 디렉토리 관리 — `get_upload_dir()`, `get_work_dir()`, `get_result_dir()`, `get_zip_path()`, `save_upload()` |
| `__init__.py` | 빈 패키지 초기화 |

## For AI Agents

### Working In This Directory
- job_manager는 DB가 아닌 파일 기반 JSON 저장소 사용 (간단한 설계)
- 작업 상태: `pending` → `processing` → `completed` / `failed`
- ⚠️ `update_job()`은 비원자적 read-modify-write — 동시 호출 시 데이터 유실 가능 (실제로는 PipelineTracker가 주요 동시 작성자)
- file_manager는 3개 볼륨 경로 관리: `uploads_volume`, `data_volume`, `result_volume`
- ZIP 파일명 형식: `result/{job_id}_{timestamp}.zip`

### Common Patterns
- 모듈 레벨 함수 (클래스 없음) — `job_manager.create_job(...)` 형태로 호출
- 경로 관리: `Path(settings.xxx_volume) / job_id` 패턴
- `save_upload()`만 async (파일 쓰기), 나머지는 동기 함수

## Dependencies

### Internal
- `app.config.settings` — 볼륨 경로 설정

### External
- `pathlib.Path`, `json`, `uuid`

<!-- MANUAL: -->
