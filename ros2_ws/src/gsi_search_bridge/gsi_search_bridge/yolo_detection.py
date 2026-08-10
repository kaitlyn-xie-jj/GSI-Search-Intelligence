"""Dependency-light adapters around Ultralytics result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .color_detection import color_image_array


@dataclass(frozen=True)
class YoloRegion:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


def image_as_bgr(image: object) -> np.ndarray:
    """Convert a sensor_msgs-like color image to contiguous BGR8."""

    pixels = color_image_array(image)
    encoding = str(image.encoding).lower()
    if encoding in {"rgba8", "bgra8"}:
        pixels = pixels[..., :3]
    if encoding in {"rgb8", "rgba8"}:
        pixels = pixels[..., ::-1]
    return np.ascontiguousarray(pixels)


def best_vehicle_region(
    results: Iterable[object],
    allowed_class_ids: Sequence[int],
    class_names: Optional[Mapping[int, str]] = None,
) -> Optional[YoloRegion]:
    """Extract the highest-confidence allowed box from Ultralytics results."""

    allowed = {int(item) for item in allowed_class_ids}
    names = class_names or {}
    regions = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = _tolist(getattr(boxes, "xyxy", ()))
        confidences = _tolist(getattr(boxes, "conf", ()))
        classes = _tolist(getattr(boxes, "cls", ()))
        for coords, confidence, class_id in zip(xyxy, confidences, classes):
            normalized_class = int(class_id)
            if normalized_class not in allowed or len(coords) < 4:
                continue
            regions.append(YoloRegion(
                x_min=float(coords[0]),
                y_min=float(coords[1]),
                x_max=float(coords[2]),
                y_max=float(coords[3]),
                confidence=float(confidence),
                class_id=normalized_class,
                class_name=str(names.get(normalized_class, normalized_class)),
            ))
    return max(regions, key=lambda item: item.confidence, default=None)


def _tolist(value: object):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
