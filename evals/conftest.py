import os
import warnings
from pathlib import Path

import pytest

from cardio_rag.config import Settings
from cardio_rag.wiring import build_pipeline
from evals import snapshot
from evals.cases import load_goldens


def replay_pipeline(path: str) -> snapshot.ReplayRag:
    """The recorded answers in the snapshot at `path`, standing in for the pipeline.

    Warns, naming the snapshot, so the scores of a replayed run are never mistaken for a live one.
    """
    recorded = snapshot.load(Path(path))
    problems = snapshot.mismatches(recorded, load_goldens())
    if problems:
        pytest.fail(
            f"EVAL_ANSWERS={path} does not match the dataset:\n" + "\n".join(problems),
            pytrace=False,
        )
    warnings.warn(
        f"EVAL_ANSWERS={path}: replaying recorded answers, not asking the live app. These scores "
        "measure the judge on fixed answers and say nothing about the current build.",
        pytest.PytestWarning,
        stacklevel=2,
    )
    return snapshot.ReplayRag(recorded)


@pytest.fixture(scope="session")
def rag():
    """One pipeline per run. The index is rebuilt so it always matches the corpus on disk.

    With EVAL_ANSWERS=<snapshot file> the recorded answers are replayed instead, so repeated runs
    score identical answers and any spread comes from the judge. For measurement only: the pull
    request gate never sets it.
    """
    path = os.environ.get("EVAL_ANSWERS")
    if path:
        return replay_pipeline(path)
    return build_pipeline(Settings.from_env(), rebuild=True)
