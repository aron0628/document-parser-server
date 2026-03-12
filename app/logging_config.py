"""중앙 로깅 설정 모듈"""

import logging
import sys

from app.config import settings


def setup_logging() -> None:
    """서버 시작 시 로깅 설정을 초기화한다."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # app 네임스페이스 로거 설정
    app_logger = logging.getLogger("app")
    app_logger.setLevel(log_level)
    # 기존 핸들러 중복 방지
    if not app_logger.handlers:
        app_logger.addHandler(handler)
