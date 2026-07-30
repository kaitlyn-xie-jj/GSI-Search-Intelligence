"""Dependency-light color target baseline for simulator interface tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class YellowThresholds:
    minimum_red: int = 150
    minimum_green: int = 120
    maximum_blue: int = 120
    minimum_yellow_margin: int = 60
    maximum_red_green_difference: int = 100
    minimum_pixels: int = 400


@dataclass(frozen=True)
class ColorRegion:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    pixel_count: int
    confidence: float

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


def find_yellow_region(image: object, thresholds: YellowThresholds) -> Optional[ColorRegion]:
    """Return one yellow region from an RGB/BGR sensor_msgs-like Image."""

    pixels = color_image_array(image)
    encoding = str(image.encoding).lower()
    if encoding in {"rgb8", "rgba8"}:
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    elif encoding in {"bgr8", "bgra8"}:
        blue, green, red = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    else:
        raise ValueError(f"unsupported color image encoding: {image.encoding}")

    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    blue_i = blue.astype(np.int16)
    mask = (
        (red_i >= thresholds.minimum_red)
        & (green_i >= thresholds.minimum_green)
        & (blue_i <= thresholds.maximum_blue)
        & (((red_i + green_i) // 2 - blue_i) >= thresholds.minimum_yellow_margin)
        & (np.abs(red_i - green_i) <= thresholds.maximum_red_green_difference)
    )
    y_values, x_values = np.nonzero(mask)
    count = int(x_values.size)
    if count < thresholds.minimum_pixels:
        return None

    x_min, x_max = int(x_values.min()), int(x_values.max())
    y_min, y_max = int(y_values.min()), int(y_values.max())
    box_area = max(1, (x_max - x_min + 1) * (y_max - y_min + 1))
    fill_ratio = count / box_area
    size_score = min(1.0, count / max(1, thresholds.minimum_pixels * 4))
    confidence = min(0.99, 0.55 + 0.25 * fill_ratio + 0.20 * size_score)
    return ColorRegion(x_min, y_min, x_max, y_max, count, confidence)


def color_image_array(image: object) -> np.ndarray:
    encoding = str(image.encoding).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise ValueError(f"unsupported color image encoding: {image.encoding}")
    rows = np.ndarray(
        shape=(int(image.height), int(image.step)),
        dtype=np.uint8,
        buffer=image.data,
    )
    active = rows[:, :int(image.width) * channels]
    return active.reshape(int(image.height), int(image.width), channels)


def depth_image_array(image: object) -> np.ndarray:
    if str(image.encoding).upper() != "32FC1":
        raise ValueError(f"unsupported depth image encoding: {image.encoding}")
    dtype = np.dtype(">f4" if bool(image.is_bigendian) else "<f4")
    return np.ndarray(
        shape=(int(image.height), int(image.width)),
        dtype=dtype,
        buffer=image.data,
        strides=(int(image.step), dtype.itemsize),
    )


def median_depth(
    image: object,
    u: float,
    v: float,
    *,
    window_radius_px: int = 4,
    minimum_depth_m: float = 0.2,
    maximum_depth_m: float = 19.1,
) -> Optional[float]:
    depth = depth_image_array(image)
    center_u, center_v = int(round(u)), int(round(v))
    u0, u1 = max(0, center_u - window_radius_px), min(depth.shape[1], center_u + window_radius_px + 1)
    v0, v1 = max(0, center_v - window_radius_px), min(depth.shape[0], center_v + window_radius_px + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    values = depth[v0:v1, u0:u1]
    valid = values[np.isfinite(values) & (values >= minimum_depth_m) & (values <= maximum_depth_m)]
    return float(np.median(valid)) if valid.size else None


def remap_pixel(
    pixel: Tuple[float, float],
    source_info: object,
    target_info: object,
) -> Tuple[float, float]:
    source_fx, source_fy = float(source_info.k[0]), float(source_info.k[4])
    source_cx, source_cy = float(source_info.k[2]), float(source_info.k[5])
    target_fx, target_fy = float(target_info.k[0]), float(target_info.k[4])
    target_cx, target_cy = float(target_info.k[2]), float(target_info.k[5])
    if min(source_fx, source_fy, target_fx, target_fy) <= 0:
        raise ValueError("camera focal lengths must be positive")
    normalized_x = (pixel[0] - source_cx) / source_fx
    normalized_y = (pixel[1] - source_cy) / source_fy
    return (
        normalized_x * target_fx + target_cx,
        normalized_y * target_fy + target_cy,
    )


def camera_point_from_pixel(
    pixel: Tuple[float, float],
    depth_m: float,
    camera_info: object,
) -> Tuple[float, float, float]:
    """Convert pinhole pixels to Gazebo camera axes: X forward, Y left, Z up."""

    fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
    cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
    if min(fx, fy, depth_m) <= 0 or not math.isfinite(depth_m):
        raise ValueError("focal lengths and depth must be finite and positive")
    right = (pixel[0] - cx) / fx * depth_m
    down = (pixel[1] - cy) / fy * depth_m
    return (depth_m, -right, -down)
