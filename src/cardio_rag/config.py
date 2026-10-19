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


# Metric key -> the judge model that metric uses when JUDGE_MODEL_<KEY> is not set. Empty: every
# metric uses judge_model(). A trial adds an entry only for a metric whose judge it replaced.
DEFAULT_JUDGE_MODELS: dict[str, str] = {}


def judge_model_for(metric_key: str) -> str:
    """The judge model for one metric key: the environment variable JUDGE_MODEL_<KEY> in capitals
    (JUDGE_MODEL_FAITHFULNESS), else its entry in DEFAULT_JUDGE_MODELS, else judge_model().
    A variable that is empty or only spaces counts as not set."""
    chosen = os.environ.get(f"JUDGE_MODEL_{metric_key.upper()}", "").strip()
    return chosen or DEFAULT_JUDGE_MODELS.get(metric_key) or judge_model()


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
