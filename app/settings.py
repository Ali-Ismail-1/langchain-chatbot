import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Provider + OpenAI (optional)
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Ollama
    ollama_model: str = "llama3:8b"
    ollama_base_url: str | None = None  # default http://localhost:11434

    # Embeddings & Chroma
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "app/data/chroma"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()