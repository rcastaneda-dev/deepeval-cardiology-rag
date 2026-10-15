"""pytest parameters for the golden cases, with the `smoke` marker applied where flagged."""

import pytest

from evals.cases import CATEGORIES, load_goldens


def all_case_params() -> list:
    return [
        pytest.param(golden, id=golden.id, marks=pytest.mark.smoke if golden.smoke else ())
        for category in CATEGORIES
        for golden in load_goldens(category)
    ]
