from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.8-27b"
    llm_api_key: str = "not-needed"
    llm_timeout: float = 120.0

    embed_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"

    db_path: str = "data/bank.sqlite3"
    checkpoints_path: str = "data/checkpoints.sqlite3"
    index_path: str = "data/policies.sqlite3"
    policies_dir: str = "knowledge/policies"

    max_tool_steps: int = 6  # сколько раз агент может сходить в тулз


@lru_cache
def settings() -> Settings:
    return Settings()
