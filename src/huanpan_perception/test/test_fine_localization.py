"""Geometry tests for YOLO-assisted fine localization."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from huanpan_perception.fine_localization import (
    CameraModel,
    PoseEstimate,
    fit_depth_plane,
    load_references,
    object_corners,
    pose_correction,
    refine_rectangle,
    save_reference,
    solve_planar_pose,
)
from huanpan_perception.pose_filter import PoseStabilizer


def camera() -> CameraModel:
    return CameraModel(
        640,
        480,
        np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        np.zeros((5, 1)),
    )


def test_ippe_recovers_synthetic_planar_pose():
    model = camera()
    expected_rvec = np.array([0.08, -0.12, 0.04])
    expected_translation = np.array([0.01, -0.005, 0.42])
    pixels, _ = cv2.projectPoints(
        object_corners(0.1, 0.025),
        expected_rvec,
        expected_translation,
        model.k,
        model.d,
    )

    pose = solve_planar_pose(pixels.reshape(4, 2), model)

    assert pose.translation == pytest.approx(expected_translation, abs=1e-7)
    assert pose.reprojection_error_px < 1e-6
    assert np.linalg.norm(pose.quaternion_xyzw) == pytest.approx(1.0)


def test_gradient_refinement_follows_projected_rectangle():
    model = camera()
    pixels, _ = cv2.projectPoints(
        object_corners(0.1, 0.025),
        np.array([0.04, -0.08, 0.03]),
        np.array([0.0, 0.0, 0.4]),
        model.k,
        model.d,
    )
    expected = pixels.reshape(4, 2)
    image = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.polylines(image, [np.rint(expected).astype(np.int32)], True, (5, 5, 5), 5)
    minimum = expected.min(axis=0)
    maximum = expected.max(axis=0)

    result = refine_rectangle(image, (*minimum, *maximum))

    assert result.method == "gradient_lines"
    assert result.quality > 0.5
    # The synthetic edge is five pixels thick, so either physical contrast edge
    # is a valid result; the fitted quad must remain within that stroke.
    assert np.max(np.linalg.norm(result.corners - expected, axis=1)) < 6.0


def _pose(translation, rvec=(0.0, 0.0, 0.0)) -> PoseEstimate:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return PoseEstimate(
        translation=np.asarray(translation, dtype=np.float64),
        rotation=rotation,
        rvec=np.asarray(rvec, dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        reprojection_error_px=0.1,
        alternate_solutions=1,
    )


def test_pose_correction_points_camera_toward_taught_pose():
    current = _pose([0.10, -0.02, 0.50])
    desired = _pose([0.0, 0.0, 0.30])
    linear, angular = pose_correction(current, desired)
    assert linear == pytest.approx([0.10, -0.02, 0.20])
    assert angular == pytest.approx([0.0, 0.0, 0.0])


def test_pose_stabilizer_requires_confirmation_and_holds_short_dropouts():
    stabilizer = PoseStabilizer(confirm_frames=3, lost_frames=3)

    assert stabilizer.update(_pose([0.0, 0.0, 0.4])).state == "ACQUIRING"
    assert stabilizer.update(_pose([0.001, 0.0, 0.4])).state == "ACQUIRING"
    tracked = stabilizer.update(_pose([0.002, 0.0, 0.4]))
    assert tracked.state == "TRACKING"
    assert tracked.tracking

    assert stabilizer.miss("test").state == "HOLDING"
    assert stabilizer.miss("test").state == "HOLDING"
    lost = stabilizer.miss("test")
    assert lost.state == "LOST"
    assert lost.pose is None


def test_pose_stabilizer_rejects_large_single_frame_jump():
    stabilizer = PoseStabilizer(
        confirm_frames=1,
        lost_frames=3,
        max_translation_jump_m=0.02,
        max_rotation_jump_deg=5.0,
    )
    assert stabilizer.update(_pose([0.0, 0.0, 0.4])).tracking

    rejected = stabilizer.update(_pose([0.1, 0.0, 0.4]))

    assert rejected.state == "HOLDING"
    assert not rejected.measurement_valid
    assert rejected.reason == "pose_jump"
    assert rejected.translation_jump_m == pytest.approx(0.1)


def test_pose_stabilizer_requires_consistent_frames_while_acquiring():
    stabilizer = PoseStabilizer(
        confirm_frames=3,
        lost_frames=3,
        max_translation_jump_m=0.02,
    )
    assert stabilizer.update(_pose([0.0, 0.0, 0.4])).state == "ACQUIRING"

    rejected = stabilizer.update(_pose([0.08, 0.0, 0.4]))

    assert rejected.state == "LOST"
    assert rejected.reason == "pose_jump"
    assert not rejected.measurement_valid


def test_reference_file_keeps_multiple_slots(tmp_path):
    path = tmp_path / "references.yaml"
    save_reference(path, 0, _pose([0.0, 0.0, 0.30]))
    save_reference(path, 11, _pose([0.1, 0.0, 0.32]))
    references = load_references(path)
    assert sorted(references) == [0, 11]
    assert references[11].translation == pytest.approx([0.1, 0.0, 0.32])


def test_depth_plane_uses_outside_faceplate_not_slot_interior():
    model = camera()
    corners = np.array(
        [[240.0, 210.0], [400.0, 210.0], [400.0, 270.0], [240.0, 270.0]]
    )
    # The cabinet faceplate is at 0.4 m, while everything visible through the
    # empty slot is deliberately placed at 0.7 m.  Sampling any part of the
    # green polygon would therefore return the wrong plane.
    depth = np.full((480, 640), 400, dtype=np.uint16)
    cv2.fillConvexPoly(depth, np.rint(corners).astype(np.int32), 700)

    plane = fit_depth_plane(depth, corners, model)

    assert plane is not None
    assert plane.median_depth_m == pytest.approx(0.4, abs=1e-4)
    assert plane.distance_m == pytest.approx(0.4, abs=1e-3)
    assert plane.inlier_ratio > 0.95
    assert plane.rmse_m < 1e-4


def test_depth_plane_ignores_invalid_pixels_inside_slot():
    model = camera()
    corners = np.array(
        [[240.0, 210.0], [400.0, 210.0], [400.0, 270.0], [240.0, 270.0]]
    )
    depth = np.full((480, 640), 400, dtype=np.uint16)
    cv2.fillConvexPoly(depth, np.rint(corners).astype(np.int32), 0)

    plane = fit_depth_plane(depth, corners, model)

    assert plane is not None
    assert plane.valid_ratio > 0.95
    assert plane.median_depth_m == pytest.approx(0.4, abs=1e-4)
