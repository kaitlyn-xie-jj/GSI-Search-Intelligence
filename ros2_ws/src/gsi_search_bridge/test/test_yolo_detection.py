from types import SimpleNamespace

import numpy as np

from gsi_search_bridge.yolo_detection import best_vehicle_region, image_as_bgr


def test_image_as_bgr_converts_rgb_and_row_padding():
    image = SimpleNamespace(
        encoding="rgb8",
        height=1,
        width=2,
        step=8,
        data=bytes([255, 0, 0, 0, 255, 0, 99, 99]),
    )
    converted = image_as_bgr(image)
    assert converted.shape == (1, 2, 3)
    assert converted.tolist() == [[[0, 0, 255], [0, 255, 0]]]
    assert converted.flags["C_CONTIGUOUS"]


def test_best_vehicle_region_ignores_non_vehicle_and_uses_best_score():
    boxes = SimpleNamespace(
        xyxy=np.array([[1, 2, 10, 20], [3, 4, 30, 40], [5, 6, 50, 60]]),
        conf=np.array([0.99, 0.55, 0.80]),
        cls=np.array([0, 2, 7]),
    )
    region = best_vehicle_region(
        [SimpleNamespace(boxes=boxes)],
        [2, 5, 7],
        {2: "car", 7: "truck"},
    )
    assert region is not None
    assert region.class_id == 7
    assert region.class_name == "truck"
    assert region.centroid == (27.5, 33.0)


def test_best_vehicle_region_returns_none_without_allowed_box():
    boxes = SimpleNamespace(
        xyxy=np.array([[1, 2, 10, 20]]),
        conf=np.array([0.99]),
        cls=np.array([0]),
    )
    assert best_vehicle_region([SimpleNamespace(boxes=boxes)], [2]) is None
