"""서버 설정 모듈"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """서버 설정"""

    # API Keys (fallback defaults - per-request headers take precedence)
    upstage_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    xai_api_key: str = ""

    # Server
    port: int = 9997
    run_mode: str = "api"
    log_level: str = "INFO"

    # Volume paths
    data_volume: str = "./data"
    result_volume: str = "./result"
    uploads_volume: str = "./uploads"

    # Embedding
    default_embedding_model: str = "embedding-passage"
    embedding_batch_size: int = 100
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "document_parser"
    db_user: str = "parser"
    db_password: str = "parser"

    # Checkpointer
    enable_checkpointer: bool = True

    # Vision
    vision_model: str = "openai/gpt-4o"

    # RAPTOR
    raptor_max_levels: int = 3
    raptor_cluster_dim: int = 5
    raptor_cluster_threshold: float = 0.3
    raptor_summarization_model: str = "openai/gpt-4.1-mini"
    min_chunks_for_raptor: int = 10
    raptor_timeout_seconds: int = 600
    raptor_max_concurrency: int = 20
    raptor_max_clusters_per_level: int = 50

    # Keyword extraction
    kiwi_num_workers: int = 1
    keyword_pos_whitelist: list[str] = ["NNG", "NNP", "SL", "SH"]
    keyword_min_length: int = 2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
