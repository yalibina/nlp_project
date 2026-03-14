from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    telegram_token: str
    llm_api_key: str = ""
    rag_collection: str = "default"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
