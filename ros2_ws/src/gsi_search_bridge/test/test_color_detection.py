import struct
import unittest
from types import SimpleNamespace

import numpy as np

from gsi_search_bridge.color_detection import (
    YellowThresholds,
    camera_point_from_pixel,
    find_yellow_region,
    median_depth,
    remap_pixel,
)


def _rgb_image(pixels):
    pixels = np.asarray(pixels, dtype=np.uint8)
    return SimpleNamespace(
        encoding="rgb8",
        height=pixels.shape[0],
        width=pixels.shape[1],
        step=pixels.shape[1] * 3,
        data=pixels.tobytes(),
    )


class ColorDetectionTests(unittest.TestCase):
    def test_finds_yellow_region(self):
        pixels = np.zeros((10, 12, 3), dtype=np.uint8)
        pixels[2:8, 3:10] = (230, 190, 20)
        region = find_yellow_region(
            _rgb_image(pixels),
            YellowThresholds(minimum_pixels=20),
        )
        self.assertIsNotNone(region)
        self.assertEqual((region.x_min, region.y_min, region.x_max, region.y_max), (3, 2, 9, 7))
        self.assertGreater(region.confidence, 0.5)

    def test_rejects_region_below_pixel_threshold(self):
        pixels = np.zeros((5, 5, 3), dtype=np.uint8)
        pixels[0, 0] = (255, 220, 0)
        self.assertIsNone(find_yellow_region(
            _rgb_image(pixels),
            YellowThresholds(minimum_pixels=2),
        ))

    def test_depth_and_pixel_geometry(self):
        values = np.full((5, 5), 4.0, dtype=np.float32)
        image = SimpleNamespace(
            encoding="32FC1",
            is_bigendian=False,
            height=5,
            width=5,
            step=20,
            data=values.tobytes(),
        )
        self.assertEqual(median_depth(image, 2, 2, window_radius_px=1), 4.0)
        info = SimpleNamespace(k=(100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0))
        self.assertEqual(camera_point_from_pixel((2.0, 2.0), 4.0, info), (4.0, -0.0, -0.0))

    def test_remaps_normalized_pixel_between_cameras(self):
        source = SimpleNamespace(k=(200.0, 0.0, 100.0, 0.0, 200.0, 50.0, 0.0, 0.0, 1.0))
        target = SimpleNamespace(k=(100.0, 0.0, 50.0, 0.0, 100.0, 25.0, 0.0, 0.0, 1.0))
        self.assertEqual(remap_pixel((120.0, 70.0), source, target), (60.0, 35.0))


if __name__ == "__main__":
    unittest.main()
