<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-17 | Updated: 2026-03-17 -->

# utils

## Purpose
PDF 분할 및 ZIP 패키징 유틸리티. 파이프라인의 전처리(PDF 분할)와 후처리(결과 ZIP) 기능을 제공한다.

## Key Files

| File | Description |
|------|-------------|
| `pdf_utils.py` | `split_pdf()` — PyPDF2로 PDF를 batch_size 페이지 단위로 분할. `test_page` 파라미터로 처리 범위 제한 가능 |
| `zip_utils.py` | `create_result_zip()` — 결과 파일(HTML, MD, CSV, PKL, images)을 ZIP으로 패키징. 구조: `{job_id}/{job_id}_{base_name}.ext` |
| `__init__.py` | 빈 패키지 초기화 |

## For AI Agents

### Working In This Directory
- `split_pdf()`는 동기 함수 (PyPDF2가 동기 전용)
- ZIP 내부 구조: `{job_id}/` 최상위 디렉토리 아래에 결과 파일 배치, images는 `images/` 하위
- ZIP 파일명: `{job_id}_{timestamp}.zip` → `result/` 디렉토리에 저장

## Dependencies

### Internal
- `app.services.file_manager.get_result_dir` (zip_utils)

### External
- `PyPDF2` (PdfReader, PdfWriter), `zipfile`, `pathlib`

<!-- MANUAL: -->
