"""
Configuration file for RAG Movie Chatbot
Update these values with your PostgreSQL credentials
"""

import os
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Database configuration class"""
    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"  # Change to your database name
    user: str = "postgres"      # Change to your username  
    password: str = ""          # Add your password here

# You can also use environment variables for security
def get_db_config_from_env():
    """Get database config from environment variables"""
    return DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

# OpenAI configuration (for future use)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384 