import math

from gsi_search_bridge.search_node import _horizontal_position_uncertainty_m


def test_horizontal_position_uncertainty_uses_xy_covariance():
    covariance = [0.0] * 36
    covariance[0] = 9.0
    covariance[7] = 16.0

    assert _horizontal_position_uncertainty_m(covariance) == 5.0


def test_horizontal_position_uncertainty_clamps_negative_variance():
    covariance = [0.0] * 36
    covariance[0] = -1.0
    covariance[7] = 4.0

    assert math.isclose(_horizontal_position_uncertainty_m(covariance), 2.0)
