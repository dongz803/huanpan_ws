"""MuJoCo workcell wrapper that produces labelled slot observations.

The scene owns one ``mujoco.MjModel``/``MjData`` pair for the huanpan workcell,
knows where the simulated perception camera and the twelve rack bays are, and
can:
  * randomise the rack pose, the camera pose and the lighting,
  * render RGB and metric depth from the perception camera,
  * project the known slot geometry into the image to produce detection labels
    and the ground-truth slot poses that the motion module will consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mujoco

from .camera import (
    Intrinsics,
    look_at_quat,
    matrix_to_quat,
    project,
    quat_to_matrix,
)

# --- Rack constants, mirrored from
# src/huanpan_description/mujoco/universal_robots_ur5e/sa5212_empty_rack.xml ---
#
# The rack shell and the front trim grid are flush at x = -0.0875 in rack-local
# coordinates, which is the plane a camera actually sees. Slot openings are
# 100 mm x 25 mm (from the drive_bay_*_recess geoms) and the full front outline
# is 430 mm x 85 mm.
RACK_FRONT_X = -0.0875
SLOT_HALF_Y = 0.050
SLOT_HALF_Z = 0.0125
RACK_HALF_Y = 0.215
RACK_HALF_Z = 0.0425

DEFAULT_MODEL = (
    "src/huanpan_description/mujoco/universal_robots_ur5e/workcell.xml"
)


@dataclass
class Detection:
    """One labelled object in one image."""

    class_id: int
    class_name: str
    slot: int
    xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    depth_m: float
    entry_position: tuple[float, float, float]
    entry_quat_wxyz: tuple[float, float, float, float]

    @property
    def width_px(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height_px(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


class SlotWorkcell:
    """Simulated RGB-D observation source for the rack slots."""

    SLOT_CLASS_ID = 0
    GRID_CLASS_ID = 1
    CLASS_NAMES = ("slot", "slot_grid")

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        camera_name: str = "perception_cam",
        rack_body_name: str = "sa5212_chassis",
        width: int = 640,
        height: int = 480,
        keyframe: str = "home",
    ) -> None:
        self.model_path = str(model_path)
        self.width = int(width)
        self.height = int(height)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.camera_id = self._require(mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.rack_body_id = self._require(mujoco.mjtObj.mjOBJ_BODY, rack_body_name)
        self.keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe
        )
        if self.keyframe_id < 0:
            raise ValueError(f"unknown MuJoCo keyframe: {keyframe}")

        self.intrinsics = Intrinsics.from_fovy(
            self.width, self.height, float(self.model.cam_fovy[self.camera_id])
        )

        self._recess_ids = {}
        for slot in range(12):
            self._recess_ids[slot] = self._require(
                mujoco.mjtObj.mjOBJ_GEOM, f"drive_bay_{slot:02d}_recess"
            )
        self._entry_site_ids = {
            slot: self._require(mujoco.mjtObj.mjOBJ_SITE, f"slot_{slot:02d}_entry")
            for slot in range(12)
        }

        # Base camera pose from the MJCF, kept as the centre of the
        # randomisation range. The look-at target is the rack front centre.
        self.base_cam_pos = np.array(self.model.cam_pos[self.camera_id], dtype=float)
        self.base_cam_quat = np.array(self.model.cam_quat[self.camera_id], dtype=float)
        self.base_light_pos = np.array(self.model.light_pos[0], dtype=float)
        self.base_light_dir = np.array(self.model.light_dir[0], dtype=float)
        self.base_light_diffuse = np.array(self.model.light_diffuse[0], dtype=float)

        self._renderer: mujoco.Renderer | None = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def _require(self, obj_type, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"{name!r} not found in {self.model_path}")
        return obj_id

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "SlotWorkcell":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def reset(self) -> None:
        """Restore the nominal scene: home pose, MJCF camera and lighting."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        self.model.cam_pos[self.camera_id] = self.base_cam_pos
        self.model.cam_quat[self.camera_id] = self.base_cam_quat
        self.model.light_pos[0] = self.base_light_pos
        self.model.light_dir[0] = self.base_light_dir
        self.model.light_diffuse[0] = self.base_light_diffuse
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------- randomisation
    def randomize(self, rng: np.random.Generator, level: str = "train") -> dict:
        """Perturb rack pose, camera pose and lighting.

        ``level`` selects the perturbation magnitude: ``"train"`` is a moderate
        working range, ``"hard"`` is a wider range used as a held-out
        generalisation split.
        """
        if level == "train":
            pos_xy, pos_z = 0.012, 0.006
            yaw, pitch, roll = np.radians(2.0), np.radians(1.0), np.radians(1.0)
            cam_xy, cam_z = 0.030, 0.020
            cam_yaw, cam_pitch, cam_roll = (
                np.radians(4.0),
                np.radians(3.0),
                np.radians(3.0),
            )
            light_shift, light_jitter = 0.30, 0.25
        elif level == "hard":
            pos_xy, pos_z = 0.025, 0.012
            yaw, pitch, roll = np.radians(6.0), np.radians(3.0), np.radians(3.0)
            cam_xy, cam_z = 0.060, 0.040
            cam_yaw, cam_pitch, cam_roll = (
                np.radians(9.0),
                np.radians(7.0),
                np.radians(6.0),
            )
            light_shift, light_jitter = 0.55, 0.45
        else:
            raise ValueError(f"unknown randomisation level: {level}")

        # --- rack pose ---
        rack_pos = self.model.body_pos[self.rack_body_id].copy()
        rack_pos[0] += rng.uniform(-pos_xy, pos_xy)
        rack_pos[1] += rng.uniform(-pos_xy, pos_xy)
        rack_pos[2] += rng.uniform(-pos_z, pos_z)
        self.model.body_pos[self.rack_body_id] = rack_pos
        rack_euler = np.array(
            [
                rng.uniform(-roll, roll),
                rng.uniform(-pitch, pitch),
                rng.uniform(-yaw, yaw),
            ]
        )
        self.model.body_quat[self.rack_body_id] = _euler_to_quat(rack_euler)

        # --- camera pose: offset in the base camera frame, then re-aim ---
        base_rotation = quat_to_matrix(self.base_cam_quat)
        offset = base_rotation @ np.array(
            [
                rng.uniform(-cam_xy, cam_xy),
                rng.uniform(-cam_xy, cam_xy),
                rng.uniform(-cam_z, cam_z),
            ]
        )
        cam_pos = self.base_cam_pos + offset
        # Aim at the (possibly moved) rack front centre with an extra angular
        # perturbation applied to the target.
        rack_rotation = quat_to_matrix(self.model.body_quat[self.rack_body_id])
        front_center = rack_pos + rack_rotation @ np.array([RACK_FRONT_X, 0.0, 0.0])
        target = front_center + np.array(
            [
                0.0,
                rng.uniform(-0.06, 0.06) * (cam_xy / 0.03),
                rng.uniform(-0.05, 0.05) * (cam_z / 0.02),
            ]
        )
        self.model.cam_pos[self.camera_id] = cam_pos
        self.model.cam_quat[self.camera_id] = look_at_quat(cam_pos, target)

        # Extra in-plane roll so the horizon is not always level.
        roll_angle = rng.uniform(-cam_roll, cam_roll)
        if abs(roll_angle) > 1e-6:
            rotation = quat_to_matrix(self.model.cam_quat[self.camera_id])
            rotation = rotation @ _rot_x(roll_angle)
            self.model.cam_pos[self.camera_id] = cam_pos
            self.model.cam_quat[self.camera_id] = matrix_to_quat(rotation)

        # --- lighting ---
        self.model.light_pos[0] = self.base_light_pos + rng.uniform(
            -light_shift, light_shift, size=3
        )
        self.model.light_dir[0] = self.base_light_dir + rng.uniform(
            -light_jitter, light_jitter, size=3
        )
        self.model.light_diffuse[0] = np.clip(
            self.base_light_diffuse + rng.uniform(-0.18, 0.18, size=3), 0.05, 1.0
        )

        mujoco.mj_forward(self.model, self.data)
        return {
            "rack_pos": rack_pos.tolist(),
            "rack_quat": self.model.body_quat[self.rack_body_id].tolist(),
            "camera_pos": np.asarray(self.model.cam_pos[self.camera_id]).tolist(),
            "camera_quat": np.asarray(self.model.cam_quat[self.camera_id]).tolist(),
            "light_pos": np.asarray(self.model.light_pos[0]).tolist(),
            "light_dir": np.asarray(self.model.light_dir[0]).tolist(),
        }

    # ----------------------------------------------------------------- render
    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        return self._renderer

    def render_rgb(self) -> np.ndarray:
        renderer = self._ensure_renderer()
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=self.camera_id)
        return renderer.render().copy()

    def render_depth(self) -> np.ndarray:
        renderer = self._ensure_renderer()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=self.camera_id)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        return depth

    # ------------------------------------------------------------- geometry
    def camera_pose(self):
        return (
            np.array(self.model.cam_pos[self.camera_id], dtype=float),
            np.array(self.model.cam_quat[self.camera_id], dtype=float),
        )

    def slot_local_center(self, slot: int) -> np.ndarray:
        """Slot centre in rack-local coordinates, from the recess geom."""
        return np.array(self.model.geom_pos[self._recess_ids[slot]], dtype=float)

    def slot_openings_world(self, slot: int) -> np.ndarray:
        """Four world-space corners of the visible slot opening."""
        return self._rack_local_to_world(self._opening_corners_local(slot))

    def grid_outline_world(self) -> np.ndarray:
        """Four world-space corners of the rack front outline."""
        local = np.array(
            [
                [RACK_FRONT_X, -RACK_HALF_Y, -RACK_HALF_Z],
                [RACK_FRONT_X, RACK_HALF_Y, -RACK_HALF_Z],
                [RACK_FRONT_X, RACK_HALF_Y, RACK_HALF_Z],
                [RACK_FRONT_X, -RACK_HALF_Y, RACK_HALF_Z],
            ]
        )
        return self._rack_local_to_world(local)

    def _opening_corners_local(self, slot: int) -> np.ndarray:
        center = self.slot_local_center(slot)
        y, z = center[1], center[2]
        return np.array(
            [
                [RACK_FRONT_X, y - SLOT_HALF_Y, z - SLOT_HALF_Z],
                [RACK_FRONT_X, y + SLOT_HALF_Y, z - SLOT_HALF_Z],
                [RACK_FRONT_X, y + SLOT_HALF_Y, z + SLOT_HALF_Z],
                [RACK_FRONT_X, y - SLOT_HALF_Y, z + SLOT_HALF_Z],
            ]
        )

    def _rack_local_to_world(self, points: np.ndarray) -> np.ndarray:
        rotation = quat_to_matrix(self.model.body_quat[self.rack_body_id])
        origin = np.array(self.model.body_pos[self.rack_body_id], dtype=float)
        return points @ rotation.T + origin

    def slot_entry_pose(self, slot: int) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the slot entry site (insertion +x of the rack frame)."""
        site_id = self._entry_site_ids[slot]
        return (
            np.array(self.data.site_xpos[site_id], dtype=float),
            np.array(self.data.site_xmat[site_id], dtype=float).reshape(3, 3),
        )

    # --------------------------------------------------------------- labels
    def sample_detections(
        self,
        include_grid: bool = True,
        min_area_px: float = 120.0,
        margin_px: float = 2.0,
    ) -> list[Detection]:
        """Ground-truth boxes for the current randomised scene.

        Boxes whose projected outline leaves the image or becomes too small are
        dropped so that labels never describe a truncated or invisible slot.
        """
        cam_pos, cam_quat = self.camera_pose()
        detections: list[Detection] = []

        for slot in range(12):
            corners = self.slot_openings_world(slot)
            uv, depth = project(corners, cam_pos, cam_quat, self.intrinsics)
            if not np.all(np.isfinite(depth)):
                continue
            x0, y0 = uv.min(axis=0)
            x1, y1 = uv.max(axis=0)
            if (
                x0 < margin_px
                or y0 < margin_px
                or x1 > self.width - margin_px
                or y1 > self.height - margin_px
            ):
                continue
            if (x1 - x0) * (y1 - y0) < min_area_px:
                continue
            entry_pos, entry_rot = self.slot_entry_pose(slot)
            detections.append(
                Detection(
                    class_id=self.SLOT_CLASS_ID,
                    class_name=self.CLASS_NAMES[self.SLOT_CLASS_ID],
                    slot=slot,
                    xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    center_px=(float(uv[:, 0].mean()), float(uv[:, 1].mean())),
                    depth_m=float(np.mean(depth)),
                    entry_position=tuple(float(v) for v in entry_pos),
                    entry_quat_wxyz=tuple(
                        float(v) for v in matrix_to_quat(entry_rot)
                    ),
                )
            )

        if include_grid:
            uv, depth = project(
                self.grid_outline_world(), cam_pos, cam_quat, self.intrinsics
            )
            if np.all(np.isfinite(depth)):
                x0, y0 = uv.min(axis=0)
                x1, y1 = uv.max(axis=0)
                x0 = max(x0, 0.0)
                y0 = max(y0, 0.0)
                x1 = min(x1, float(self.width - 1))
                y1 = min(y1, float(self.height - 1))
                if (x1 - x0) * (y1 - y0) >= min_area_px:
                    detections.append(
                        Detection(
                            class_id=self.GRID_CLASS_ID,
                            class_name=self.CLASS_NAMES[self.GRID_CLASS_ID],
                            slot=-1,
                            xyxy=(float(x0), float(y0), float(x1), float(y1)),
                            center_px=(float(uv[:, 0].mean()), float(uv[:, 1].mean())),
                            depth_m=float(np.mean(depth)),
                            entry_position=(0.0, 0.0, 0.0),
                            entry_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                        )
                    )
        return detections


# ------------------------------------------------------------------ helpers
def _euler_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
    """MuJoCo ``wxyz`` quaternion from roll/pitch/yaw applied as Rz@Ry@Rx."""
    rx, ry, rz = (float(v) for v in euler_xyz)
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ]
    )


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
