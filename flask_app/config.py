"""
App configuration and environment loader.
Supports dual-backend: Google Gemini API (primary) + OpenRouter (fallback).
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-12345")
    # Google Gemini API (primary - 1500 free req/day)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # OpenRouter API (fallback)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
    OPENROUTER_EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
    # Redirect writable folders to /tmp on Vercel's read-only environment
    if os.environ.get("VERCEL"):
        UPLOAD_FOLDER = "/tmp/uploads"
        OUTPUT_FOLDER = "/tmp/outputs"
    else:
        UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
        OUTPUT_FOLDER = os.path.join(os.getcwd(), "outputs")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limits
    POPPLER_PATH = os.getenv("POPPLER_PATH", None)

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY and not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "Neither GEMINI_API_KEY nor OPENROUTER_API_KEY is set. "
                "Please add at least one to your .env file."
            )
