"""Central configuration loader.

Reads `config.yaml` (tunable knobs) and `.env` (secrets) once, and exposes them
as a single `cfg` object the rest of the code imports. Keeping all config in one
place means you tune the system by editing YAML, not by hunting through .py files.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv

# Project root = two levels up from this file (src/config.py -> project/)
ROOT = Path(__file__).resolve().parent.parent

# Load secrets from .env into environment variables
load_dotenv(ROOT / ".env")


class Config:
    """Holds the merged YAML config plus secrets from the environment."""

    def __init__(self, yaml_path: Path):
        with open(yaml_path, "r") as f:
            self._raw = yaml.safe_load(f)

        # --- secrets (from .env) ---
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "") or None

    # Convenience accessors so callers write cfg.embedding["model_name"] etc.
    def __getattr__(self, name):
        # Only called if normal attribute lookup fails -> look in YAML
        raw = self.__dict__.get("_raw", {})
        if name in raw:
            return raw[name]
        raise AttributeError(f"No config key '{name}'")


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached singleton — config is read from disk only once per process."""
    return Config(ROOT / "config.yaml")


# Importable shortcut: `from src.config import cfg`
cfg = get_config()


if __name__ == "__main__":
    # Quick sanity check: python -m src.config
    print("Project root:", ROOT)
    print("arXiv categories:", cfg.arxiv["categories"])
    print("Embedding model:", cfg.embedding["model_name"])
    print("Groq model:", cfg.generation["model_name"])
    print("HF token set:", bool(cfg.hf_token))
    print("Groq key set:", bool(cfg.groq_api_key))
    print("Qdrant URL:", cfg.qdrant_url)
