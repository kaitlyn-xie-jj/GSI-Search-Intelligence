import math
import struct
import unittest
from types import SimpleNamespace

from gsi_search_bridge.pointcloud_projection import PointCloudGroundProjector


def _cloud(points, frame_id="camera_link"):
    data = b"".join(struct.pack("<fff", *point) for point in points)
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        width=len(points),
        height=1,
        fields=(
            SimpleNamespace(name="x", offset=0, datatype=7),
            SimpleNamespace(name="y", offset=4, datatype=7),
            SimpleNamespace(name="z", offset=8, datatype=7),
        ),
        point_step=12,
        row_step=12 * len(points),
        is_bigendian=False,
        data=data,
    )


def _pose(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


class PointCloudProjectionTests(unittest.TestCase):
    def test_filters_non_ground_and_non_finite_points(self):
        projector = PointCloudGroundProjector(
            ground_plane_z_m=0.0,
            ground_tolerance_m=0.1,
            point_resolution_m=0.1,
            expected_frame_id="camera_link",
        )
        result = projector.project(
            _cloud(((1.0, 2.0, 0.0), (2.0, 3.0, 0.05), (3.0, 4.0, 1.0),
                    (math.nan, 0.0, 0.0))),
            _pose(),
        )
        self.assertEqual(result, ((1.0, 2.0), (2.0, 3.0)))

    def test_applies_camera_pitch_and_body_translation(self):
        projector = PointCloudGroundProjector(
            camera_translation=(0.0, 0.0, 0.0),
            camera_rpy=(0.0, math.pi / 2.0, 0.0),
            ground_plane_z_m=1.0,
            ground_tolerance_m=0.01,
            point_resolution_m=0.1,
        )
        result = projector.project(_cloud(((1.0, 0.0, 0.0),)), _pose(4.0, 5.0, 2.0))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][0], 4.0)
        self.assertAlmostEqual(result[0][1], 5.0)

    def test_deduplicates_points_at_configured_resolution(self):
        projector = PointCloudGroundProjector(
            ground_tolerance_m=0.1,
            point_resolution_m=0.5,
        )
        result = projector.project(
            _cloud(((1.01, 2.01, 0.0), (1.02, 2.02, 0.0))),
            _pose(),
        )
        self.assertEqual(len(result), 1)

    def test_rejects_unexpected_frame(self):
        projector = PointCloudGroundProjector(expected_frame_id="camera_link")
        with self.assertRaisesRegex(ValueError, "does not match"):
            projector.project(_cloud(((1.0, 2.0, 0.0),), "wrong_frame"), _pose())


if __name__ == "__main__":
    unittest.main()
