.PHONY: install run test lint docker-up docker-down docker-build clean health

# 의존성 설치
install:
	uv sync --dev

# 로컬 서버 실행
run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 9997

# 로컬 서버 실행 (자동 리로드)
dev:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 9997 --reload

# 테스트 실행
test:
	uv run pytest

# 린트
lint:
	uv run ruff check app/

# 포맷팅
format:
	uv run black app/

# Docker 빌드 + 실행
docker-up:
	docker compose up --build -d

# Docker 중지
docker-down:
	docker compose down

# Docker 빌드만
docker-build:
	docker compose build

# Docker 로그 확인
docker-logs:
	docker compose logs -f

# 서버 헬스체크
health:
	@curl -s http://localhost:9997/health | python3 -m json.tool

# 작업 목록 확인
jobs:
	@curl -s http://localhost:9997/jobs | python3 -m json.tool

# 임시 파일 정리
clean:
	rm -rf data/jobs/* uploads/* result/* __pycache__ .pytest_cache
