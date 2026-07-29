"""
Application configuration.

Reads settings from environment variables (via a local .env file in
development). No secrets are hard-coded — copy .env.example to .env
and fill in real values to use a hosted LLM provider.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Provider selection: "openai" or "local".
    # "local" runs fully offline with a deterministic embedding + an
    # extractive answerer, so the project can be cloned and run with
    # zero API keys for demo/testing purposes.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    DOCS_DIR: str = os.getenv("DOCS_DIR", "data/sample_docs")
    INDEX_DIR: str = os.getenv("INDEX_DIR", "data/index")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))


settings = Settings()
