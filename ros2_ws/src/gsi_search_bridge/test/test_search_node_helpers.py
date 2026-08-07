import math
from types import SimpleNamespace

import pytest

from gsi_search_bridge.search_node import (
    _can_use_transit_detection_evidence,
    _belief_total_variation,
    _horizontal_position_uncertainty_m,
    _negative_update_rejection_reason,
    _new_visible_cell_count,
    _projection_visibility_probability,
    _quality_after_sensor_skew,
    _replan_reason,
    _rgb_image_to_ppm,
    _transit_replan_protected,
    _transit_suspect_cell_is_available,
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


def test_projection_visibility_requires_enough_real_ground_points():
    assert _projection_visibility_probability(0, 10) == 0.0
    assert _projection_visibility_probability(5, 10) == 0.5
    assert _projection_visibility_probability(20, 10) == 1.0


def test_negative_update_rejection_reports_observation_failure():
    base = {
        "rgb_available": True,
        "depth_available": True,
        "point_cloud_available": True,
        "projected_ground_point_count": 10,
        "minimum_projected_ground_points": 10,
        "observation_quality": 1.0,
        "minimum_observation_quality": 0.5,
    }
    assert _negative_update_rejection_reason(**base) is None
    assert _negative_update_rejection_reason(
        **{**base, "point_cloud_available": False}
    ) == "no_valid_point_projection"
    assert _negative_update_rejection_reason(
        **{**base, "projected_ground_point_count": 2}
    ) == "blocked_view"


def test_replan_hysteresis_suppresses_small_frequent_updates():
    params = {
        "positive_detection": False,
        "belief_total_variation": 0.02,
        "kl_divergence_nats": 0.01,
        "trajectory_valid": True,
        "expected_reward_change": 0.05,
        "time_since_replan_s": 5.0,
        "minimum_interval_s": 20.0,
        "belief_total_variation_threshold": 0.1,
        "kl_divergence_threshold_nats": 0.05,
        "expected_reward_change_threshold": 0.2,
    }
    assert _replan_reason(**params) is None
    assert _replan_reason(
        **{**params, "positive_detection": True}
    ) == "positive_detection"
    assert _replan_reason(
        **{
            **params,
            "time_since_replan_s": 25.0,
            "kl_divergence_nats": 0.08,
        }
    ) == "kl_divergence_exceeded"


def test_transit_suspect_inspection_is_not_interrupted_by_blocked_frames():
    assert _transit_replan_protected(
        trigger="negative_detection_in_transit",
        verification_in_progress=False,
        suspect_inspection_in_progress=True,
    )


def test_positive_detection_interrupts_transit_suspect_inspection():
    assert not _transit_replan_protected(
        trigger="positive_detection_in_transit",
        verification_in_progress=False,
        suspect_inspection_in_progress=True,
    )


def test_verification_remains_protected_from_transit_replans():
    assert _transit_replan_protected(
        trigger="positive_detection_in_transit",
        verification_in_progress=True,
        suspect_inspection_in_progress=False,
    )


def test_transit_suspect_cell_stops_replanning_after_global_limit():
    counts = {"cell-a": 2}

    assert not _transit_suspect_cell_is_available("cell-a", counts, 2)
    assert _transit_suspect_cell_is_available("cell-b", counts, 2)
    assert not _transit_suspect_cell_is_available("cell-b", counts, 0)


def test_belief_total_variation_is_symmetric():
    first = {"a": 0.8, "b": 0.2}
    second = {"a": 0.5, "b": 0.5}
    assert _belief_total_variation(first, second) == pytest.approx(0.3)
    assert _belief_total_variation(second, first) == pytest.approx(0.3)


def _detection(confidence=0.9, localization_error_m=1.0):
    return SimpleNamespace(
        confidence=confidence,
        attributes={"localization_error_m": localization_error_m},
    )


def _observation(*detections):
    return SimpleNamespace(detections=detections)


def test_transit_detection_is_allowed_after_prior_negative_viewpoints():
    assert _can_use_transit_detection_evidence(
        [_observation(), _observation()],
        [_detection()],
        minimum_confidence=0.5,
        maximum_localization_error_m=8.0,
    )


def test_transit_detection_rejects_low_quality_current_evidence():
    assert not _can_use_transit_detection_evidence(
        [_observation()],
        [_detection(confidence=0.4), _detection(localization_error_m=9.0)],
        minimum_confidence=0.5,
        maximum_localization_error_m=8.0,
    )


def test_transit_detection_does_not_supply_second_positive_confirmation():
    assert not _can_use_transit_detection_evidence(
        [_observation(_detection())],
        [_detection()],
        minimum_confidence=0.5,
        maximum_localization_error_m=8.0,
    )


def test_new_visible_cell_count_only_counts_quality_improvements():
    assert _new_visible_cell_count(
        {"cell-a": 0.7, "cell-b": 0.4},
        ("cell-a", "cell-b", "cell-b", "cell-c"),
        0.6,
    ) == 2


def test_rgb_image_to_ppm_converts_bgr_and_ignores_row_padding():
    image = SimpleNamespace(
        width=1,
        height=1,
        step=4,
        encoding="bgr8",
        data=bytes((3, 2, 1, 99)),
    )

    assert _rgb_image_to_ppm(image) == b"P6\n1 1\n255\n\x01\x02\x03"
