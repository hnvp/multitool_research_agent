import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "qwen3:8b",
    )

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    tavily_api_key: str = os.getenv(
        "TAVILY_API_KEY",
        "",
    )

    openweather_api_key: str = os.getenv(
        "OPENWEATHER_API_KEY",
        "",
    )

    pinecone_api_key: str = os.getenv(
        "PINECONE_API_KEY",
        "",
    )

    pinecone_index: str = os.getenv(
        "PINECONE_INDEX",
        "multitool-agent-rag",
    )

    rag_embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "intfloat/multilingual-e5-base",
    )

    rag_reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3",
    )

    rag_top_k: int = int(os.getenv("RAG_TOP_K", "20"))

    rag_top_n: int = int(os.getenv("RAG_TOP_N", "4"))


settings = Settings()
