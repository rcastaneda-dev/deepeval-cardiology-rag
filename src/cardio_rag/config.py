"""Environment-driven settings. Keys are read lazily so offline code never needs them."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
INDEX_PATH = REPO_ROOT / "data" / "index.npz"

DISCLAIMER = (
    "This is a software demo that answers questions from a small set of US government "
    "health documents. It is not medical advice. Talk to a clinician about your own "
    "health, and in an emergency call 911 (US)."
)

load_dotenv(REPO_ROOT / ".env")


class MissingSettingError(RuntimeError):
    """A required environment variable is not set."""


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSettingError(
            f"Environment variable {name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class Settings:
    generator_model: str
    embedding_model: str
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            generator_model=os.environ.get("GENERATOR_MODEL", "claude-haiku-5-5"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            top_k=int(os.environ.get("TOP_K", "4")),
        )
