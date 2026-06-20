from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # Database Configuration
    DB_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/sentinai",
        description="Async connection string for PostgreSQL."
    )
    
    # S3 / AWS Configuration
    AWS_REGION: str = Field(default="us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # OpenLineage Configuration
    OPENLINEAGE_URL: Optional[str] = Field(
        default=None,
        description="URL for the OpenLineage backend (e.g. Marquez)."
    )
    OPENLINEAGE_NAMESPACE: str = Field(
        default="sentinai-ingestion",
        description="OpenLineage namespace for tracking."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

config = Settings()
