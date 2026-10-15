import pytest

from cardio_rag.config import Settings
from cardio_rag.wiring import build_pipeline


@pytest.fixture(scope="session")
def rag():
    """One pipeline per run. The index is rebuilt so it always matches the corpus on disk."""
    return build_pipeline(Settings.from_env(), rebuild=True)
