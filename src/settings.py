from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_batch_size: int = 15
    llm_temperature: float = 0.0
    llm_confidence_threshold: float = 0.6
    enable_llm_cache: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
