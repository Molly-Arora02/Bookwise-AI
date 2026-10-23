"""
BookWise AI — Configuration and Settings Module.

Manages application paths, chunking parameters, retrieval thresholds,
LLM settings, and UI design tokens.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
DEFAULT_PDF_PATH = DATA_DIR / "History_2024-25.pdf"


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    pdf_path: Path = DEFAULT_PDF_PATH
    chroma_persist_dir: Path = CHROMA_PERSIST_DIR
    collection_name: str = "bookwise_history_collection"

    # Ingestion & Chunking parameters
    chunk_size: int = Field(default=800, description="Chunk size in characters")
    chunk_overlap: int = Field(default=150, description="Chunk overlap in characters")

    # Retrieval parameters
    top_k: int = Field(default=5, description="Number of candidate chunks to retrieve")
    retrieval_threshold: float = Field(
        default=0.42,
        description="Max cosine distance threshold for candidate relevance (lower = more similar)",
    )
    neighbor_window: int = Field(
        default=1,
        description="Number of neighboring chunks before/after to retrieve for context continuity",
    )

    # Model & Generation parameters
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Default local embedding model",
    )
    llm_provider: str = Field(
        default="local",
        description="LLM provider: 'local' (deterministic grounded extractor), 'gemini', 'openai'",
    )
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API Key (optional, can be passed in .env or UI)",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key (optional)",
    )

    # Design System Tokens — Strictly Light, Warm Palette (NO BLACK OR BLUE BACKGROUNDS)
    ui_theme: dict = {
        # Backgrounds
        "bg_main": "#faf8f5",          # Warm ivory background
        "bg_sidebar": "#f5f0e6",       # Soft cream sidebar
        "bg_card": "#ffffff",          # Crisp white card surface
        "bg_card_subtle": "#fbf9f6",   # Light warm stone surface
        "bg_accent_soft": "#f3ede3",   # Warm beige accent surface
        
        # Primary accents (Sage green, soft olive, terracotta)
        "accent_sage": "#556b4f",      # Muted sage green
        "accent_sage_light": "#eaf0e8",# Very soft sage tint
        "accent_terracotta": "#c86446",# Elegant terracotta
        "accent_terracotta_soft": "#faece8", # Subtle peach/terracotta tint
        "accent_olive": "#646f4b",     # Soft olive
        
        # Borders & Dividers
        "border_light": "#e8e1d7",     # Soft warm sand border
        "border_subtle": "#f0ebe1",    # Very subtle border
        "border_accent": "#c5baaa",    # Muted taupe border
        
        # Typography (Warm charcoal, deep brown, muted taupe)
        "text_primary": "#2b2826",     # Warm dark charcoal (high readability, NOT blue/black)
        "text_secondary": "#5a534c",   # Warm taupe
        "text_muted": "#857c74",       # Muted stone gray
        "text_accent": "#8c432d",      # Deep terracotta text accent
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings singleton
settings = Settings()
