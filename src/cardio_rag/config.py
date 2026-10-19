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


# Metric key -> the judge model that metric uses when JUDGE_MODEL_<KEY> is not set; a metric with
# no entry uses judge_model(). Both Faithfulness checks use gpt-5-mini, chosen by the trial of
# 2026-10-19 on the 54 faithfulness and bias_faithfulness answers of evals/snapshots/2026-10-19.json
# (3 judgings of each, the same answers for every judge). "Listed" is the nine answers that had a
# FLAKY faithfulness entry; "fell" is an answer that passed 3 of 3 under gpt-4o-mini and did not;
# "caught" is the contradicts_source answers (in-013, in-018, in-024) that failed most judgings.
#   judge          listed passing 3 of 3   fell                 caught   cost
#   gpt-4o-mini    6 of 9                  -                    2 of 3   $0.14
#   gpt-4.1-mini   9 of 9                  in-028 (0 of 3)      3 of 3   $0.41
#   gpt-5-mini     9 of 9                  none                 3 of 3   $1.36
# gpt-4.1-mini was rejected for in-028. Its answer restates the passage and the golden, but the
# judge scored it 0.50 in all 3 judgings: it read "(regurgitation)" as the name of the leak alone,
# so "valve flaps ... the wrong size or shape, so the valve leaks" was not a definition of it, and
# its extracted truths left out the sentence defining stenosis. The abstention answers that give
# a fact from memory (oos-001, 002, 003) passed Faithfulness under every judge, because
# Faithfulness only looks for contradictions of the context; the abstention rubric catches them.
# gpt-4.1 was not run, as gpt-5-mini qualified first. gpt-5-mini costs about 10 times as much per
# judging as gpt-4o-mini ($0.0076 against $0.0008) and its 180 judgings took 12 minutes.
DEFAULT_JUDGE_MODELS: dict[str, str] = {
    "faithfulness": "gpt-5-mini",
    "bias_faithfulness": "gpt-5-mini",
}


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
