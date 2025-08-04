"""
Configuration file for RAG Movie Chatbot
"""

import os

# OpenAI configuration (for future use)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# File paths
DATA_DIR = "data"
SUBTITLE_DIR = os.path.join(DATA_DIR, "subtitles")
INDEX_DIR = os.path.join(DATA_DIR, "index")