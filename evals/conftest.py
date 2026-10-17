import os
from pathlib import Path

import pytest

from cardio_rag.config import Settings
from cardio_rag.wiring import build_pipeline
from evals import snapshot
from evals.cases import load_goldens


@pytest.fixture(scope="session")
def rag():
    """One pipeline per run. The index is rebuilt so it always matches the corpus on disk.

    With EVAL_ANSWERS=<snapshot file> the recorded answers are replayed instead, so repeated runs
    score identical answers and any spread comes from the judge. For measurement only: the pull
    request gate never sets it.
    """
    path = os.environ.get("EVAL_ANSWERS")
    if path:
        recorded = snapshot.load(Path(path))
        problems = snapshot.mismatches(recorded, load_goldens())
        if problems:
            pytest.fail(
                f"EVAL_ANSWERS={path} does not match the dataset:\n" + "\n".join(problems),
                pytrace=False,
            )
        return snapshot.ReplayRag(recorded)
    return build_pipeline(Settings.from_env(), rebuild=True)
