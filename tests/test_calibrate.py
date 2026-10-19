import pytest

from evals.calibrate import propose_threshold


def test_threshold_keeps_a_margin_below_the_lowest_acceptable_score():
    assert propose_threshold([0.83, 0.97]) == (0.75, [], [])


def test_the_floor_wins_and_reports_the_margin_risk():
    assert propose_threshold([0.54]) == (0.5, [0.54], [])


def test_the_ceiling_caps_the_threshold():
    assert propose_threshold([1.0]) == (0.9, [], [])


def test_an_exact_multiple_is_not_lost_to_float_error():
    # 0.6 - 0.05 is 0.5499999999999999, which a plain floor would round down to 0.5.
    assert propose_threshold([0.6]) == (0.55, [], [])


def test_every_score_inside_the_margin_is_listed_in_the_order_given():
    # Floor 0.5: 0.52 and 0.54 sit under 0.05 above it, 0.56 does not.
    assert propose_threshold([0.54, 0.9, 0.52, 0.56]) == (0.5, [0.54, 0.52], [])


def test_the_margin_and_bounds_can_be_changed():
    assert propose_threshold([0.83], margin=0.1, floor=0.6, ceiling=0.7) == (0.7, [], [])
    assert propose_threshold([0.62], margin=0.1, floor=0.6, ceiling=0.7) == (0.6, [0.62], [])


def test_an_acceptable_score_below_the_floor_is_listed_as_failing_not_as_a_margin_risk():
    # The floor wins, so 0.4 fails the 0.5 threshold; only 0.8 passes it, with room to spare.
    assert propose_threshold([0.4, 0.8]) == (0.5, [], [0.4])


def test_failing_scores_are_listed_in_the_order_given_apart_from_the_margin_risks():
    # 0.3 and 0.45 fail the 0.5 floor, 0.52 passes it by less than the margin, 0.7 is clear.
    assert propose_threshold([0.45, 0.52, 0.7, 0.3]) == (0.5, [0.52], [0.45, 0.3])


def test_no_acceptable_scores_is_an_error_not_a_threshold():
    with pytest.raises(ValueError, match="acceptable"):
        propose_threshold([])
