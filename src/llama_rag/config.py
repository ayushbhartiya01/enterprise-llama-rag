from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    cohere_api_key: str
    llama_cloud_api_key: str = Field(
        validation_alias=AliasChoices("LLAMA_CLOUD_API_KEY", "LLAMAPARSE_API_KEY")
    )
    qdrant_url: str
    
    # Model Configurations
    llm_model: str = "openai/gpt-oss-120b"
    embed_model_name: str = "BAAI/bge-large-en-v1.5"
    
    # Hyperparameters
    similarity_top_k: int = 20
    rerank_top_n: int = 3
    rerank_model: str = "rerank-v5.0-pro"
    grounding_threshold: float = 0.4
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
