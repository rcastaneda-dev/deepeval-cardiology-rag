"""Record the app's answer to every golden question, so the judge can be measured on its own.

Usage: uv run python scripts/record_answers.py [--out PATH]

Rebuilds the index, asks the app (Claude generates) each golden question once, and writes the
answers, their citations and the retrieved passages to evals/snapshots/<today>.json, or to PATH.
It makes paid embedding and generation calls and is not part of the quality gate. Replay a
snapshot with `EVAL_ANSWERS=<file> uv run deepeval test run evals/`: the gate never does.
"""

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

from cardio_rag.config import Settings
from cardio_rag.wiring import build_pipeline

# Run as a file (`python scripts/record_answers.py`), so make the repo root importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evals.cases import load_goldens  # noqa: E402
from evals.snapshot import record, save  # noqa: E402

SNAPSHOT_DIR = REPO_ROOT / "evals" / "snapshots"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="where to write (default: evals/snapshots/<date>)")
    args = parser.parse_args(argv)

    today = date.today().isoformat()
    out = args.out or SNAPSHOT_DIR / f"{today}.json"
    settings = Settings.from_env()
    meta = {
        "created": today,
        "git_sha": _git_sha(),
        "generator_model": settings.generator_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
    }
    snapshot = record(load_goldens(), build_pipeline(settings, rebuild=True), meta=meta)
    save(snapshot, out)
    print(f"Wrote {len(snapshot['cases'])} answers to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
