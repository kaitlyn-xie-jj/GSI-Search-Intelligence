import math

import pytest

from gsi_search_bridge.search_node import (
    _horizontal_position_uncertainty_m,
    _quality_after_sensor_skew,
)


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


def test_quality_after_sensor_skew_matches_adapter_decay():
    assert _quality_after_sensor_skew(0.8, 0.0, 2.0) == 0.8
    assert _quality_after_sensor_skew(0.8, 0.5, 2.0) == pytest.approx(0.6)
    assert _quality_after_sensor_skew(0.8, 3.0, 2.0) == 0.0


def test_quality_after_sensor_skew_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="maximum_sensor_skew_s"):
        _quality_after_sensor_skew(0.8, 0.1, 0.0)
