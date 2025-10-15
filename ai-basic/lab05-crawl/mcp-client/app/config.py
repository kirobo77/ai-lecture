"""Configuration management for ARI Processing Client"""
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv

# lab05 디렉토리의 .env 파일 로드
lab05_root = Path(__file__).parent.parent.parent
env_path = lab05_root / ".env"
load_dotenv(env_path)

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "info"
    
    # MCP Server Configuration
    mcp_server_url: str = "http://127.0.0.1:4200/my-custom-path/"
    mcp_connection_timeout: int = 30
    mcp_retry_attempts: int = 3
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_allow_credentials: bool = True
    
    # Application Configuration
    app_title: str = "ARI Processing Server"
    app_version: str = "1.0.0"
    
    # AI & LLM Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    
    # Vector Database Configuration
    qdrant_host: str = ""
    
    # Search Engine Configuration
    opensearch_host: str = ""
    
    # Database Configuration
    database_url: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # 추가 필드 무시

# Global settings instance
settings = Settings()