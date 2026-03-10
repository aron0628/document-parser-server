"""서버 설정 모듈"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """서버 설정"""

    # API Keys (fallback defaults - per-request headers take precedence)
    upstage_api_key: str = ""
    openai_api_key: str = ""

    # Server
    port: int = 9997
    run_mode: str = "api"
    max_upload_size_mb: int = 100

    # Volume paths
    data_volume: str = "./data"
    result_volume: str = "./result"
    uploads_volume: str = "./uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
