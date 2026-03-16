"""서버 설정 모듈"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """서버 설정"""

    # API Keys (fallback defaults - per-request headers take precedence)
    upstage_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Server
    port: int = 9997
    run_mode: str = "api"
    log_level: str = "INFO"
    max_upload_size_mb: int = 100

    # Volume paths
    data_volume: str = "./data"
    result_volume: str = "./result"
    uploads_volume: str = "./uploads"

    # Embedding
    default_embedding_model: str = "embedding-passage"
    embedding_batch_size: int = 100

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "document_parser"
    db_user: str = "parser"
    db_password: str = "parser"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
