"""Projection and labelling contract for the simulated perception camera."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from slotvision.camera import Intrinsics, look_at_quat, project, unproject
from slotvision.scene import RACK_FRONT_X, SlotWorkcell

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODEL = (
    WORKSPACE_ROOT
    / "src"
    / "huanpan_description"
    / "mujoco"
    / "universal_robots_ur5e"
    / "workcell.xml"
)

# This simulation bootstrap is parked: the project switched to a real D435 and
# the simulated `perception_cam` was removed from the shared workcell model.
# See perception/README.md. Re-add the camera to workcell.xml to re-enable.
_CAMERA_PRESENT = MODEL.is_file() and "perception_cam" in MODEL.read_text(
    encoding="utf-8"
)
pytestmark = pytest.mark.skipif(
    not _CAMERA_PRESENT,
    reason=(
        "simulation bootstrap is parked: workcell.xml has no perception_cam "
        "(the project now uses a real D435; see perception/README.md)"
    ),
)


@pytest.fixture(scope="module")
def scene():
    with SlotWorkcell(model_path=str(MODEL)) as workcell:
        yield workcell


def test_intrinsics_match_d435_vertical_fov(scene: SlotWorkcell) -> None:
    intrinsics = scene.intrinsics
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    assert intrinsics.fx == pytest.approx(617.16, abs=0.5)
    assert intrinsics.cx == pytest.approx(320.0)
    assert intrinsics.cy == pytest.approx(240.0)
    assert intrinsics.hfov_deg == pytest.approx(54.8, abs=0.5)


def test_slot_09_projects_to_expected_pixel(scene: SlotWorkcell) -> None:
    scene.reset()
    corners = scene.slot_openings_world(9)
    uv, depth = project(
        corners, *scene.camera_pose(), scene.intrinsics
    )
    assert np.all(np.isfinite(depth))
    center = uv.mean(axis=0)
    assert center[0] == pytest.approx(255.1, abs=2.0)
    assert center[1] == pytest.approx(280.0, abs=2.0)
    assert float(depth.mean()) == pytest.approx(0.50, abs=0.01)


def test_projected_depth_matches_rendered_depth(scene: SlotWorkcell) -> None:
    scene.reset()
    depth_image = scene.render_depth()
    for slot in (0, 5, 11):
        uv, depth = project(
            scene.slot_openings_world(slot), *scene.camera_pose(), scene.intrinsics
        )
        u, v = uv.mean(axis=0)
        rendered = float(depth_image[int(round(v)), int(round(u))])
        # The label plane sits 5.5 mm in front of the recess front face that the
        # depth ray actually hits, so allow 20 mm of slack.
        assert rendered == pytest.approx(float(depth.mean()), abs=0.02)


def test_unproject_round_trips(scene: SlotWorkcell) -> None:
    scene.reset()
    cam_pos, cam_quat = scene.camera_pose()
    point = scene.slot_openings_world(4)[0]
    uv, depth = project(point, cam_pos, cam_quat, scene.intrinsics)
    recovered = unproject(uv[0], depth[0], cam_pos, cam_quat, scene.intrinsics)
    assert np.allclose(recovered, point, atol=1e-6)


def test_all_twelve_slots_are_labelled_and_inside_the_image(
    scene: SlotWorkcell,
) -> None:
    scene.reset()
    detections = scene.sample_detections()
    slots = sorted(d.slot for d in detections if d.slot >= 0)
    assert slots == list(range(12))

    grids = [d for d in detections if d.slot < 0]
    assert len(grids) == 1

    for det in detections:
        x0, y0, x1, y1 = det.xyxy
        assert 0 <= x0 < x1 <= scene.width
        assert 0 <= y0 < y1 <= scene.height
        assert det.depth_m > 0.1

    slot_boxes = [d for d in detections if d.slot >= 0]
    for det in slot_boxes:
        # A slot is a wide, short opening: 100 mm x 25 mm.
        assert 2.5 < det.width_px / det.height_px < 5.5


def test_slot_entry_pose_matches_the_mjcf(scene: SlotWorkcell) -> None:
    scene.reset()
    position, rotation = scene.slot_entry_pose(9)
    # Rack body origin plus the slot_09_entry site offset.
    expected = np.array([0.7075 - 0.1025, 0.0525, 0.750 - 0.0275])
    assert np.allclose(position, expected, atol=1e-6)
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-6)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-9


def test_randomisation_keeps_labels_valid_and_moves_the_geometry(
    scene: SlotWorkcell,
) -> None:
    rng = np.random.default_rng(7)
    scene.reset()
    nominal = scene.slot_openings_world(0).copy()
    moved = False
    for _ in range(12):
        scene.reset()
        scene.randomize(rng, level="train")
        detections = scene.sample_detections()
        for det in detections:
            x0, y0, x1, y1 = det.xyxy
            assert x0 >= 0 and y0 >= 0 and x1 <= scene.width and y1 <= scene.height
        if not np.allclose(scene.slot_openings_world(0), nominal, atol=1e-4):
            moved = True
    assert moved, "rack randomisation never changed the geometry"


def test_hard_split_is_wider_than_train(scene: SlotWorkcell) -> None:
    def spread(level: str, seed: int = 3) -> float:
        rng = np.random.default_rng(seed)
        offsets = []
        scene.reset()
        base = np.array(scene.model.cam_pos[scene.camera_id], dtype=float)
        for _ in range(20):
            scene.reset()
            scene.randomize(rng, level=level)
            offsets.append(np.linalg.norm(scene.model.cam_pos[scene.camera_id] - base))
        return float(np.mean(offsets))

    assert spread("hard") > spread("train")


def test_look_at_quat_points_the_camera_at_the_target() -> None:
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([1.0, 0.2, -0.3])
    quat = look_at_quat(eye, target)
    from slotvision.camera import quat_to_matrix

    rotation = quat_to_matrix(quat)
    viewing = -rotation[:, 2]
    expected = target / np.linalg.norm(target)
    assert np.allclose(viewing, expected, atol=1e-9)


def test_intrinsics_from_fovy_is_square_pixel() -> None:
    intrinsics = Intrinsics.from_fovy(640, 480, 42.5)
    assert intrinsics.fx == intrinsics.fy
    assert math.isclose(intrinsics.cx, 320.0)
    assert math.isclose(intrinsics.cy, 240.0)
