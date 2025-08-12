import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", "app/data/chroma")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()