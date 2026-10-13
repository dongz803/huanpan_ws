"""ROS 2 YOLO ROI -> sub-pixel slot corners -> planar PnP pose node."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .capture_metadata import stamp_to_ns
from .detector import DetectorError, RackDetector
from .fine_localization import (
    CameraModel,
    coarse_rectangle,
    fit_depth_plane,
    load_references,
    pose_correction,
    refine_rectangle_candidates,
    save_reference,
    select_best_planar_pose,
    solve_planar_pose,
)
from .front_view import (
    FrontViewGate,
    estimate_front_view,
    estimate_roll_from_horizontal_edges,
)
from .image_utils import array_to_rgb8_image_msg, image_msg_to_array
from .pose_filter import PoseStabilizer


DEFAULT_PARAMETERS = {
    "image_topic": "/camera/camera/color/image_raw",
    "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic": "/camera/camera/color/camera_info",
    "model_path": "",
    "conf": 0.25,
    "iou": 0.5,
    # Existing weights were trained at 640.  Edge refinement still runs on the
    # original 1280x720 RGB frame; raise this only after retraining at 1280.
    "imgsz": 640,
    "device": "cpu",
    "max_det": 20,
    "target_class": "slot_opening",
    "target_slot": 0,
    "slot_width_m": 0.100,
    "slot_height_m": 0.025,
    "refinement_mode": "edges",  # edges | bbox
    "min_refinement_quality": 0.18,
    "max_reprojection_error_px": 2.0,
    "max_sync_delta_ms": 30.0,
    "translation_tolerance_m": 0.003,
    "rotation_tolerance_deg": 1.0,
    "temporal_window_size": 5,
    "tracking_confirm_frames": 3,
    "tracking_lost_frames": 3,
    "max_translation_jump_m": 0.05,
    "max_rotation_jump_deg": 15.0,
    "depth_min_valid_ratio": 0.25,
    "depth_min_inlier_ratio": 0.60,
    "max_pnp_depth_delta_m": 0.05,
    "enforce_pnp_depth_consistency": False,
    "front_view_enabled": True,
    "front_view_only": True,
    "require_front_view_alignment": True,
    "front_view_window_size": 5,
    "front_view_confirm_frames": 5,
    "front_view_tilt_tolerance_deg": 5.0,
    "front_view_roll_tolerance_deg": 3.0,
    "front_view_exit_tilt_deg": 8.0,
    "front_view_exit_roll_deg": 5.0,
    "front_view_min_line_quality": 0.15,
    "front_view_max_plane_rmse_m": 0.010,
    "reference_file": "",
    "sync_queue_size": 15,
    "sync_slop_s": 0.03,
    "pose_topic": "/huanpan/vision/slot_pose",
    "correction_topic": "/huanpan/vision/correction",
    "aligned_topic": "/huanpan/vision/aligned",
    "debug_image_topic": "/huanpan/vision/debug_image",
    # Debug display only. Pose estimation always keeps the original RGB size.
    # A smaller best-effort preview avoids blocking the real-time callback on a
    # multi-megabyte reliable Image publish.
    "debug_max_width": 640,
    "status_topic": "/huanpan/vision/status",
    "front_view_correction_topic": "/huanpan/vision/front_view_correction",
    "front_view_aligned_topic": "/huanpan/vision/front_view_aligned",
}


class FineLocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__("huanpan_fine_localizer")
        for name, default in DEFAULT_PARAMETERS.items():
            self.declare_parameter(name, default)

        model_path = str(self.get_parameter("model_path").value)
        if not model_path:
            raise DetectorError("parameter 'model_path' is empty")
        self._detector = RackDetector(
            model_path,
            conf=float(self.get_parameter("conf").value),
            iou=float(self.get_parameter("iou").value),
            imgsz=int(self.get_parameter("imgsz").value),
            device=str(self.get_parameter("device").value),
            max_det=int(self.get_parameter("max_det").value),
        )
        self._target_class = str(self.get_parameter("target_class").value)
        self._target_slot = int(self.get_parameter("target_slot").value)
        if not 0 <= self._target_slot <= 11:
            raise ValueError("target_slot must be in [0, 11]")
        self._slot_width = float(self.get_parameter("slot_width_m").value)
        self._slot_height = float(self.get_parameter("slot_height_m").value)
        self._refinement_mode = str(self.get_parameter("refinement_mode").value)
        if self._refinement_mode not in {"edges", "bbox"}:
            raise ValueError("refinement_mode must be 'edges' or 'bbox'")
        self._min_quality = float(
            self.get_parameter("min_refinement_quality").value
        )
        self._max_reprojection = float(
            self.get_parameter("max_reprojection_error_px").value
        )
        self._max_sync_delta_ms = float(
            self.get_parameter("max_sync_delta_ms").value
        )
        self._translation_tolerance = float(
            self.get_parameter("translation_tolerance_m").value
        )
        self._rotation_tolerance = np.deg2rad(
            float(self.get_parameter("rotation_tolerance_deg").value)
        )
        self._stabilizer = PoseStabilizer(
            window_size=int(self.get_parameter("temporal_window_size").value),
            confirm_frames=int(self.get_parameter("tracking_confirm_frames").value),
            lost_frames=int(self.get_parameter("tracking_lost_frames").value),
            max_translation_jump_m=float(
                self.get_parameter("max_translation_jump_m").value
            ),
            max_rotation_jump_deg=float(
                self.get_parameter("max_rotation_jump_deg").value
            ),
        )
        self._depth_min_valid_ratio = float(
            self.get_parameter("depth_min_valid_ratio").value
        )
        self._depth_min_inlier_ratio = float(
            self.get_parameter("depth_min_inlier_ratio").value
        )
        self._max_pnp_depth_delta_m = float(
            self.get_parameter("max_pnp_depth_delta_m").value
        )
        self._enforce_pnp_depth_consistency = bool(
            self.get_parameter("enforce_pnp_depth_consistency").value
        )
        self._front_view_enabled = bool(
            self.get_parameter("front_view_enabled").value
        )
        self._front_view_only = bool(
            self.get_parameter("front_view_only").value
        )
        self._require_front_view_alignment = bool(
            self.get_parameter("require_front_view_alignment").value
        )
        self._front_view_min_line_quality = float(
            self.get_parameter("front_view_min_line_quality").value
        )
        self._front_view_max_plane_rmse_m = float(
            self.get_parameter("front_view_max_plane_rmse_m").value
        )
        self._front_view_gate = FrontViewGate(
            window_size=int(self.get_parameter("front_view_window_size").value),
            confirm_frames=int(
                self.get_parameter("front_view_confirm_frames").value
            ),
            tilt_tolerance_deg=float(
                self.get_parameter("front_view_tilt_tolerance_deg").value
            ),
            roll_tolerance_deg=float(
                self.get_parameter("front_view_roll_tolerance_deg").value
            ),
            exit_tilt_tolerance_deg=float(
                self.get_parameter("front_view_exit_tilt_deg").value
            ),
            exit_roll_tolerance_deg=float(
                self.get_parameter("front_view_exit_roll_deg").value
            ),
        )
        configured_reference = str(self.get_parameter("reference_file").value)
        self._reference_file = Path(
            configured_reference
            or "outputs/perception/references/slot_references.yaml"
        ).expanduser()
        self._references = load_references(self._reference_file)
        self._current_pose = None
        self._busy = threading.Lock()
        self._debug_max_width = int(self.get_parameter("debug_max_width").value)
        if self._debug_max_width < 0:
            raise ValueError("debug_max_width must be >= 0")

        self._pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("pose_topic").value), 10
        )
        self._correction_pub = self.create_publisher(
            TwistStamped, str(self.get_parameter("correction_topic").value), 10
        )
        self._aligned_pub = self.create_publisher(
            Bool, str(self.get_parameter("aligned_topic").value), 10
        )
        debug_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            debug_qos,
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self._front_view_correction_pub = self.create_publisher(
            TwistStamped,
            str(self.get_parameter("front_view_correction_topic").value),
            10,
        )
        self._front_view_aligned_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("front_view_aligned_topic").value),
            10,
        )
        self.create_service(Trigger, "/huanpan/vision/teach_reference", self._teach)

        image_sub = Subscriber(
            self,
            Image,
            str(self.get_parameter("image_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        depth_sub = Subscriber(
            self,
            Image,
            str(self.get_parameter("depth_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        info_sub = Subscriber(
            self,
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        self._subscribers = (image_sub, depth_sub, info_sub)
        self._sync = ApproximateTimeSynchronizer(
            list(self._subscribers),
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_s").value),
            allow_headerless=False,
        )
        self._sync.registerCallback(self._on_synced)
        self.get_logger().info(
            "fine localizer ready: YOLO ROI -> front-view alignment -> %s -> "
            "IPPE, slot=%d, reference=%s"
            % (self._refinement_mode, self._target_slot, self._reference_file)
        )

    def _teach(self, _request, response):
        if self._current_pose is None:
            response.success = False
            response.message = "no valid slot pose is available"
            return response
        save_reference(self._reference_file, self._target_slot, self._current_pose)
        self._references[self._target_slot] = self._current_pose
        response.success = True
        response.message = (
            f"saved slot {self._target_slot} reference to {self._reference_file}"
        )
        return response

    @staticmethod
    def _camera_model(msg: CameraInfo) -> CameraModel:
        return CameraModel(
            width=int(msg.width),
            height=int(msg.height),
            k=np.asarray(msg.k, dtype=np.float64).reshape(3, 3),
            d=np.asarray(msg.d, dtype=np.float64).reshape(-1, 1),
            frame_id=msg.header.frame_id,
        )

    def _publish_status(self, payload: dict) -> None:
        self._status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    @staticmethod
    def _front_view_status(front_state, plane=None, details=None) -> dict:
        measurement = front_state.measurement
        payload = {
            "state": front_state.state,
            "aligned": front_state.aligned,
            "confirm_count": front_state.confirm_count,
            "reason": front_state.reason,
        }
        if measurement is not None:
            payload.update(
                {
                    "tilt_deg": round(measurement.tilt_deg, 3),
                    "roll_deg": round(measurement.roll_deg, 3),
                    "line_quality": round(measurement.line_quality, 4),
                    "plane_normal": [
                        round(float(value), 6)
                        for value in measurement.plane_normal
                    ],
                    "correction_rotation_deg": [
                        round(float(value), 3)
                        for value in np.rad2deg(measurement.correction_rotvec)
                    ],
                }
            )
        if plane is not None:
            payload["plane"] = {
                "depth_m": round(plane.median_depth_m, 5),
                "valid_ratio": round(plane.valid_ratio, 4),
                "inlier_ratio": round(plane.inlier_ratio, 4),
                "rmse_m": round(plane.rmse_m, 5),
            }
        if details:
            payload.update(details)
        return payload

    def _publish_front_view(self, image_msg, front_state) -> None:
        self._front_view_aligned_pub.publish(Bool(data=front_state.aligned))
        measurement = front_state.measurement
        if measurement is None:
            return
        message = TwistStamped()
        message.header = image_msg.header
        (
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z,
        ) = (float(value) for value in measurement.correction_rotvec)
        self._front_view_correction_pub.publish(message)

    @staticmethod
    def _finish_timing(timing: dict[str, float], started: float) -> dict[str, float]:
        """Return stable, human-readable per-stage callback timings."""
        timing["total"] = (time.perf_counter() - started) * 1000.0
        return {name: round(float(value), 3) for name, value in timing.items()}

    def _on_synced(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if not self._busy.acquire(blocking=False):
            return
        started = time.perf_counter()
        timing = {
            "convert": 0.0,
            "yolo": 0.0,
            "edges": 0.0,
            "pnp": 0.0,
            "depth": 0.0,
            "front_view": 0.0,
            "outputs": 0.0,
            "debug_publish": 0.0,
            "total": 0.0,
        }
        frame = None
        target = None
        sync_delta_ms = None
        front_state = None
        front_plane = None
        front_details = {}
        try:
            stamps = [
                stamp_to_ns(color_msg.header.stamp),
                stamp_to_ns(depth_msg.header.stamp),
                stamp_to_ns(info_msg.header.stamp),
            ]
            sync_delta_ms = (max(stamps) - min(stamps)) / 1_000_000.0
            if sync_delta_ms > self._max_sync_delta_ms:
                temporal = self._stabilizer.miss("sync_delta")
                if self._front_view_enabled:
                    front_state = self._front_view_gate.miss("sync_delta")
                    self._publish_front_view(color_msg, front_state)
                self._current_pose = None
                self._publish_status(
                    {
                        "valid": False,
                        "reason": "sync_delta",
                        "tracking_state": temporal.state,
                        "sync_delta_ms": sync_delta_ms,
                        "timing_ms": self._finish_timing(timing, started),
                    }
                )
                return
            stage_started = time.perf_counter()
            try:
                frame = image_msg_to_array(color_msg)
                depth = image_msg_to_array(depth_msg)
                camera = self._camera_model(info_msg)
            finally:
                timing["convert"] = (time.perf_counter() - stage_started) * 1000.0
            dimensions_match = (
                frame.shape[:2] == depth.shape[:2]
                and frame.shape[1] == camera.width
                and frame.shape[0] == camera.height
            )
            if not dimensions_match:
                temporal = self._stabilizer.miss("dimension_mismatch")
                if self._front_view_enabled:
                    front_state = self._front_view_gate.miss(
                        "dimension_mismatch"
                    )
                    self._publish_front_view(color_msg, front_state)
                self._current_pose = None
                self._publish_status(
                    {
                        "valid": False,
                        "reason": "dimension_mismatch",
                        "tracking_state": temporal.state,
                        "timing_ms": self._finish_timing(timing, started),
                    }
                )
                return

            stage_started = time.perf_counter()
            try:
                detections = self._detector.detect(frame)
            finally:
                timing["yolo"] = (time.perf_counter() - stage_started) * 1000.0
            targets = [d for d in detections if d.class_name == self._target_class]
            target = self._detector.largest(targets or detections)
            if target is None:
                temporal = self._stabilizer.miss("no_detection")
                if self._front_view_enabled:
                    front_state = self._front_view_gate.miss("no_detection")
                    self._publish_front_view(color_msg, front_state)
                self._current_pose = None
                stage_started = time.perf_counter()
                self._aligned_pub.publish(Bool(data=False))
                timing["outputs"] = (time.perf_counter() - stage_started) * 1000.0
                stage_started = time.perf_counter()
                self._publish_debug(
                    color_msg,
                    frame,
                    None,
                    None,
                    temporal.pose,
                    temporal.state,
                )
                timing["debug_publish"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                self._publish_status(
                    {
                        "valid": False,
                        "reason": "no_detection",
                        "tracking_state": temporal.state,
                        "sync_delta_ms": sync_delta_ms,
                        "timing_ms": self._finish_timing(timing, started),
                    }
                )
                return

            if self._front_view_enabled:
                stage_started = time.perf_counter()
                coarse = coarse_rectangle(target.xyxy)
                try:
                    front_plane = fit_depth_plane(depth, coarse.corners, camera)
                    if front_plane is None:
                        raise ValueError("outside panel depth plane is unavailable")
                    if front_plane.valid_ratio < self._depth_min_valid_ratio:
                        raise ValueError(
                            "outside depth valid ratio %.3f is below %.3f"
                            % (
                                front_plane.valid_ratio,
                                self._depth_min_valid_ratio,
                            )
                        )
                    if front_plane.inlier_ratio < self._depth_min_inlier_ratio:
                        raise ValueError(
                            "outside depth inlier ratio %.3f is below %.3f"
                            % (
                                front_plane.inlier_ratio,
                                self._depth_min_inlier_ratio,
                            )
                        )
                    if front_plane.rmse_m > self._front_view_max_plane_rmse_m:
                        raise ValueError(
                            "outside depth plane RMSE %.4fm exceeds %.4fm"
                            % (
                                front_plane.rmse_m,
                                self._front_view_max_plane_rmse_m,
                            )
                        )
                    roll_deg, line_quality = estimate_roll_from_horizontal_edges(
                        frame, target.xyxy
                    )
                    if line_quality < self._front_view_min_line_quality:
                        raise ValueError(
                            "long-edge quality %.3f is below %.3f"
                            % (line_quality, self._front_view_min_line_quality)
                        )
                    front_measurement = estimate_front_view(
                        front_plane,
                        roll_deg,
                        line_quality=line_quality,
                    )
                    front_state = self._front_view_gate.update(front_measurement)
                    front_details = {
                        "orientation_source": "outside_depth_plane_plus_long_edge",
                        "long_edge_quality": round(line_quality, 4),
                    }
                except (ValueError, cv2.error) as exc:
                    front_state = self._front_view_gate.miss(str(exc))
                    front_details = {
                        "orientation_source": "outside_depth_plane_plus_long_edge"
                    }
                finally:
                    timing["front_view"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0

                self._publish_front_view(color_msg, front_state)
                stop_after_front_view = self._front_view_only or (
                    self._require_front_view_alignment
                    and not front_state.aligned
                )
                if stop_after_front_view:
                    temporal = self._stabilizer.miss("front_view_not_aligned")
                    self._current_pose = None
                    stage_started = time.perf_counter()
                    self._aligned_pub.publish(Bool(data=False))
                    timing["outputs"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    measurement = front_state.measurement
                    correction = (
                        None
                        if measurement is None
                        else (
                            np.zeros(3, dtype=np.float64),
                            measurement.correction_rotvec,
                        )
                    )
                    state_text = front_state.state
                    if measurement is not None:
                        state_text += " tilt %.1f roll %.1f" % (
                            measurement.tilt_deg,
                            measurement.roll_deg,
                        )
                    stage_started = time.perf_counter()
                    self._publish_debug(
                        color_msg,
                        frame,
                        target.xyxy,
                        # Orientation stage has not produced fine corners yet.
                        # Keep the YOLO box yellow and do not redraw the same
                        # coordinates as a misleading green quadrilateral.
                        None,
                        None,
                        state_text,
                        correction,
                    )
                    timing["debug_publish"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    if measurement is None:
                        reason = "front_view_unavailable"
                    elif self._front_view_only and front_state.aligned:
                        reason = "front_view_only"
                    else:
                        reason = "front_view_not_aligned"
                    self._publish_status(
                        {
                            "valid": False,
                            "pose_valid": False,
                            "reason": reason,
                            "stage": "front_view",
                            "front_view_only": self._front_view_only,
                            "coarse_confidence": round(target.confidence, 4),
                            "bbox_xyxy": [
                                round(float(value), 2) for value in target.xyxy
                            ],
                            "tracking_state": temporal.state,
                            "front_view": self._front_view_status(
                                front_state, front_plane, front_details
                            ),
                            "sync_delta_ms": round(sync_delta_ms, 3),
                            "timing_ms": self._finish_timing(timing, started),
                        }
                    )
                    return

            stage_started = time.perf_counter()
            try:
                if self._refinement_mode == "bbox":
                    rectangle = coarse_rectangle(target.xyxy)
                else:
                    rectangles = refine_rectangle_candidates(frame, target.xyxy)
                    rectangles = [
                        candidate
                        for candidate in rectangles
                        if candidate.quality >= self._min_quality
                    ]
                    if not rectangles:
                        raise ValueError(
                            "no edge quadrilateral reaches quality %.2f"
                            % self._min_quality
                        )
            finally:
                timing["edges"] = (time.perf_counter() - stage_started) * 1000.0
            stage_started = time.perf_counter()
            try:
                if self._refinement_mode == "bbox":
                    pose = solve_planar_pose(
                        rectangle.corners,
                        camera,
                        width_m=self._slot_width,
                        height_m=self._slot_height,
                    )
                else:
                    rectangle, pose = select_best_planar_pose(
                        rectangles,
                        camera,
                        width_m=self._slot_width,
                        height_m=self._slot_height,
                    )
            finally:
                timing["pnp"] = (time.perf_counter() - stage_started) * 1000.0
            if pose.reprojection_error_px > self._max_reprojection:
                raise ValueError(
                    f"reprojection error {pose.reprojection_error_px:.2f}px exceeds "
                    f"{self._max_reprojection:.2f}px"
                )
            stage_started = time.perf_counter()
            try:
                plane = fit_depth_plane(depth, rectangle.corners, camera)
            finally:
                timing["depth"] = (time.perf_counter() - stage_started) * 1000.0
            depth_delta = None
            depth_consistent = None
            if (
                plane is not None
                and plane.valid_ratio >= self._depth_min_valid_ratio
                and plane.inlier_ratio >= self._depth_min_inlier_ratio
            ):
                depth_delta = abs(
                    plane.median_depth_m - float(pose.translation[2])
                )
                depth_consistent = depth_delta <= self._max_pnp_depth_delta_m
                if (
                    not depth_consistent
                    and self._enforce_pnp_depth_consistency
                ):
                    raise ValueError(
                        f"PnP/depth difference {depth_delta:.3f}m exceeds "
                        f"{self._max_pnp_depth_delta_m:.3f}m "
                        f"(PnP Z={float(pose.translation[2]):.3f}m, "
                        f"outside-panel depth={plane.median_depth_m:.3f}m)"
                    )

            temporal = self._stabilizer.update(pose)
            if not temporal.measurement_valid:
                self._current_pose = None
                stage_started = time.perf_counter()
                self._aligned_pub.publish(Bool(data=False))
                timing["outputs"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                stage_started = time.perf_counter()
                self._publish_debug(
                    color_msg,
                    frame,
                    target.xyxy,
                    rectangle.corners,
                    temporal.pose,
                    temporal.state,
                )
                timing["debug_publish"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                self._publish_status(
                    {
                        "valid": False,
                        "reason": temporal.reason,
                        "tracking_state": temporal.state,
                        "translation_jump_m": round(
                            temporal.translation_jump_m, 6
                        ),
                        "rotation_jump_deg": round(
                            temporal.rotation_jump_deg, 3
                        ),
                        "sync_delta_ms": round(sync_delta_ms, 3),
                        "front_view": None
                        if front_state is None
                        else self._front_view_status(
                            front_state, front_plane, front_details
                        ),
                        "timing_ms": self._finish_timing(timing, started),
                    }
                )
                return
            pose = temporal.pose
            assert pose is not None
            stable = temporal.tracking
            self._current_pose = pose if stable else None
            stage_started = time.perf_counter()
            if stable:
                self._pose_pub.publish(self._pose_message(color_msg, pose))

            correction = None
            aligned = False
            desired = self._references.get(self._target_slot)
            if stable and desired is not None:
                linear, angular = pose_correction(pose, desired)
                correction = (linear, angular)
                message = TwistStamped()
                message.header = color_msg.header
                message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = (
                    float(value) for value in linear
                )
                message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = (
                    float(value) for value in angular
                )
                self._correction_pub.publish(message)
                aligned = bool(
                    np.linalg.norm(linear) <= self._translation_tolerance
                    and np.linalg.norm(angular) <= self._rotation_tolerance
                )
            self._aligned_pub.publish(Bool(data=aligned))
            timing["outputs"] = (time.perf_counter() - stage_started) * 1000.0

            payload = {
                "valid": stable,
                "reason": "" if stable else "acquiring",
                "tracking_state": temporal.state,
                "slot": self._target_slot,
                "coarse_confidence": round(target.confidence, 4),
                "refinement_method": rectangle.method,
                "refinement_quality": round(rectangle.quality, 4),
                "reprojection_error_px": round(pose.reprojection_error_px, 4),
                "translation_m": [
                    round(float(value), 6) for value in pose.translation
                ],
                "quaternion_xyzw": [
                    round(float(value), 7) for value in pose.quaternion_xyzw
                ],
                "sync_delta_ms": round(sync_delta_ms, 3),
                "reference_available": desired is not None,
                "aligned": aligned,
                "front_view": None
                if front_state is None
                else self._front_view_status(
                    front_state, front_plane, front_details
                ),
                "depth": None
                if plane is None
                else {
                    "source": "outside_faceplate_ring",
                    "median_m": round(plane.median_depth_m, 5),
                    "valid_ratio": round(plane.valid_ratio, 4),
                    "plane_distance_m": round(plane.distance_m, 5),
                    "inlier_ratio": round(plane.inlier_ratio, 4),
                    "rmse_m": round(plane.rmse_m, 5),
                    "pnp_depth_delta_m": round(
                        abs(plane.median_depth_m - float(pose.translation[2])), 5
                    ),
                    "consistent_with_pnp": depth_consistent,
                    "enforced": self._enforce_pnp_depth_consistency,
                },
            }
            if correction is not None:
                payload["correction"] = {
                    "translation_m": [round(float(v), 6) for v in correction[0]],
                    "rotation_deg": [
                        round(float(v), 4) for v in np.rad2deg(correction[1])
                    ],
                }
            stage_started = time.perf_counter()
            self._publish_debug(
                color_msg,
                frame,
                target.xyxy,
                rectangle.corners,
                pose,
                "ALIGNED" if aligned else temporal.state,
                correction,
            )
            timing["debug_publish"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            payload["timing_ms"] = self._finish_timing(timing, started)
            self._publish_status(payload)
        except (DetectorError, ValueError, cv2.error) as exc:
            temporal = self._stabilizer.miss(str(exc))
            self._current_pose = None
            stage_started = time.perf_counter()
            self._aligned_pub.publish(Bool(data=False))
            timing["outputs"] += (time.perf_counter() - stage_started) * 1000.0
            try:
                if frame is not None:
                    stage_started = time.perf_counter()
                    self._publish_debug(
                        color_msg,
                        frame,
                        target.xyxy if target else None,
                        None,
                        temporal.pose,
                        temporal.state,
                    )
                    timing["debug_publish"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
            except Exception:
                pass
            payload = {
                "valid": False,
                "reason": "localization_failed",
                "detail": str(exc),
                "tracking_state": temporal.state,
                "timing_ms": self._finish_timing(timing, started),
            }
            if sync_delta_ms is not None:
                payload["sync_delta_ms"] = round(sync_delta_ms, 3)
            if front_state is not None:
                payload["front_view"] = self._front_view_status(
                    front_state, front_plane, front_details
                )
            self._publish_status(payload)
            self.get_logger().warning(str(exc), throttle_duration_sec=2.0)
        except Exception as exc:  # noqa: BLE001 - keep live perception alive
            temporal = self._stabilizer.miss(
                f"{type(exc).__name__}: {exc}"
            )
            self._current_pose = None
            self._aligned_pub.publish(Bool(data=False))
            self._publish_status(
                {
                    "valid": False,
                    "reason": "internal_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "tracking_state": temporal.state,
                    "front_view": None
                    if front_state is None
                    else self._front_view_status(
                        front_state, front_plane, front_details
                    ),
                    "timing_ms": self._finish_timing(timing, started),
                }
            )
            self.get_logger().error(
                f"fine localization failed: {type(exc).__name__}: {exc}",
                throttle_duration_sec=2.0,
            )
        finally:
            self._busy.release()

    @staticmethod
    def _pose_message(image_msg: Image, pose) -> PoseStamped:
        message = PoseStamped()
        message.header = image_msg.header
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            float(value) for value in pose.translation
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = (float(value) for value in pose.quaternion_xyzw)
        return message

    def _publish_debug(
        self,
        image_msg: Image,
        frame: np.ndarray,
        bbox,
        corners,
        pose,
        state: str,
        correction=None,
    ) -> None:
        # Avoid allocating, drawing and serialising a full-resolution image
        # when nobody is viewing the debug topic.
        if self._debug_pub.get_subscription_count() == 0:
            return

        # The input and output are both RGB. Draw directly in that layout to
        # avoid the previous RGB->BGR->RGB pair of full-frame conversions.
        canvas = frame.copy()
        if bbox is not None:
            x0, y0, x1, y1 = np.rint(bbox).astype(int)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 210, 0), 3)
            cv2.putText(
                canvas,
                "YOLO",
                (x0, max(y0 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 210, 0),
                2,
                cv2.LINE_AA,
            )
        if corners is not None:
            polygon = np.rint(corners).astype(int)
            cv2.polylines(canvas, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
            for index, point in enumerate(polygon):
                cv2.circle(canvas, tuple(point), 4, (30, 80, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index),
                    tuple(point + (5, -5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            label_position = tuple(polygon[0] + (5, 18))
            cv2.putText(
                canvas,
                "FINE",
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        colour = (0, 255, 0) if state == "ALIGNED" else (255, 200, 0)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 74), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            state[:90],
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            colour,
            2,
        )
        if pose is not None:
            text = "XYZ [mm]  %.1f  %.1f  %.1f   reproj %.2f px" % (
                *(pose.translation * 1000.0),
                pose.reprojection_error_px,
            )
            cv2.putText(
                canvas,
                text,
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        if correction is not None:
            linear, angular = correction
            text = "MOVE [mm] %.1f %.1f %.1f   ROT [deg] %.2f %.2f %.2f" % (
                *(linear * 1000.0),
                *np.rad2deg(angular),
            )
            cv2.putText(
                canvas,
                text,
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                1,
                cv2.LINE_AA,
            )
        if self._debug_max_width and canvas.shape[1] > self._debug_max_width:
            scale = self._debug_max_width / float(canvas.shape[1])
            canvas = cv2.resize(
                canvas,
                (self._debug_max_width, int(round(canvas.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self._debug_pub.publish(
            array_to_rgb8_image_msg(
                canvas, frame_id=image_msg.header.frame_id, stamp=image_msg.header.stamp
            )
        )


def main(args=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.fine_localization_node")
    rclpy.init(args=args)
    node = None
    try:
        node = FineLocalizationNode()
        rclpy.spin(node)
    except (DetectorError, ValueError) as exc:
        print(f"[huanpan_perception] {exc}")
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
