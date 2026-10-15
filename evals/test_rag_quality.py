"""The DeepEval quality gate.

Run with `deepeval test run evals/`; add `-m smoke` for the pull-request subset.
"""

import pytest
from deepeval import assert_test

from evals.harness import build_test_case
from evals.metrics import metrics_for
from evals.params import all_case_params


@pytest.mark.parametrize("golden", all_case_params())
def test_rag_quality(golden, rag):
    assert_test(test_case=build_test_case(golden, rag), metrics=metrics_for(golden))
