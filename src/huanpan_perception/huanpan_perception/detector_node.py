"""ROS 2 node: YOLO rack/slot detection on the real D435 colour stream.

Subscribes to a ``sensor_msgs/Image`` topic (the RealSense colour image),
publishes the detections as ``vision_msgs/Detection2DArray`` and optionally an
annotated image for ``rqt_image_view``.

Run it with the perception virtualenv so Torch and Ultralytics are importable::

    source scripts/perception_env.sh
    .venv-perception/bin/python -m huanpan_perception.detector_node \
        --ros-args --params-file \
        install/huanpan_perception/share/huanpan_perception/config/perception.yaml
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .detector import DetectorError, RackDetector
from .image_utils import array_to_rgb8_image_msg, image_msg_to_array
from .viz import draw_detections

DEFAULT_PARAMETERS = {
    "image_topic": "/camera/camera/color/image_raw",
    "detection_topic": "detections",
    "annotated_topic": "detections/image",
    "model_path": "",
    "conf": 0.25,
    "iou": 0.5,
    "imgsz": 640,
    "device": "cpu",
    "max_det": 20,
    "output_mode": "vision_msgs",  # or "json" when vision_msgs is unavailable
    "publish_annotated": True,
    # Preview only: detection still runs on the original camera frame.
    "annotated_max_width": 640,
    "target_class": "slot_opening",
    "log_every_n_frames": 30,
}


class _VisionMsgs:
    """``vision_msgs`` message classes plus a version compatibility flag.

    ``vision_msgs`` changed ``Detection2D.results`` from ``ObjectHypothesis`` to
    ``ObjectHypothesisWithPose`` in 4.1. Support both without guessing.
    """

    def __init__(self) -> None:
        from vision_msgs.msg import (
            Detection2D,
            Detection2DArray,
            ObjectHypothesis,
            ObjectHypothesisWithPose,
        )

        self.array = Detection2DArray
        self.detection = Detection2D
        self.plain = ObjectHypothesis
        self.with_pose = ObjectHypothesisWithPose
        field_type = Detection2D.get_fields_and_field_types().get("results", "")
        self.uses_with_pose = "ObjectHypothesisWithPose" in field_type

    def hypothesis(self, class_id: int, score: float):
        if self.uses_with_pose:
            message = self.with_pose()
            message.hypothesis.class_id = str(class_id)
            message.hypothesis.score = float(score)
            return message
        message = self.plain()
        message.class_id = str(class_id)
        message.score = float(score)
        return message


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("huanpan_detector")
        for name, default in DEFAULT_PARAMETERS.items():
            self.declare_parameter(name, default)

        self._image_topic = self.get_parameter("image_topic").value
        self._target_class = self.get_parameter("target_class").value
        self._output_mode = self.get_parameter("output_mode").value
        self._publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self._annotated_max_width = int(
            self.get_parameter("annotated_max_width").value
        )
        if self._annotated_max_width < 0:
            raise ValueError("annotated_max_width must be >= 0")
        self._log_every = int(self.get_parameter("log_every_n_frames").value)

        model_path = self.get_parameter("model_path").value
        if not model_path:
            raise DetectorError(
                "parameter 'model_path' is empty; set it to a trained .pt file"
            )
        self.detector = RackDetector(
            model_path,
            conf=float(self.get_parameter("conf").value),
            iou=float(self.get_parameter("iou").value),
            imgsz=int(self.get_parameter("imgsz").value),
            device=self.get_parameter("device").value,
            max_det=int(self.get_parameter("max_det").value),
        )

        self._busy = threading.Lock()
        self._frames = 0
        self._dropped = 0
        self._last_log = time.monotonic()

        self._detection_pub = None
        self._json_pub = None
        self._vision = None
        if self._output_mode == "vision_msgs":
            try:
                self._vision = _VisionMsgs()
            except ImportError:
                self.get_logger().fatal(
                    "vision_msgs is not installed. Install it with:\n"
                    "  sudo apt install ros-humble-vision-msgs\n"
                    "or set the parameter output_mode:=json to publish a JSON "
                    "summary instead."
                )
                raise
            self._detection_pub = self.create_publisher(
                self._vision.array,
                self.get_parameter("detection_topic").value,
                10,
            )
        else:
            self._json_pub = self.create_publisher(
                String, self.get_parameter("detection_topic").value, 10
            )

        self._image_pub = None
        if self._publish_annotated:
            image_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._image_pub = self.create_publisher(
                Image, self.get_parameter("annotated_topic").value, image_qos
            )

        self.create_subscription(
            Image, self._image_topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"huanpan_detector ready: image={self._image_topic} "
            f"model={model_path} output={self._output_mode} "
            f"target_class={self._target_class}"
        )

    # ---------------------------------------------------------------- callback
    def _on_image(self, msg: Image) -> None:
        if not self._busy.acquire(blocking=False):
            self._dropped += 1
            return
        try:
            started = time.perf_counter()
            frame = image_msg_to_array(msg)
            converted = time.perf_counter()
            detections = self.detector.detect(frame)
            detected = time.perf_counter()
            self._publish(msg, detections, frame)
            published = time.perf_counter()
            self._frames += 1
            if self._log_every > 0 and self._frames % self._log_every == 0:
                self._log_status(
                    detections,
                    convert_ms=(converted - started) * 1000.0,
                    yolo_ms=(detected - converted) * 1000.0,
                    publish_ms=(published - detected) * 1000.0,
                )
        except DetectorError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=10.0)
        except Exception as exc:  # noqa: BLE001 - keep the node alive
            self.get_logger().error(
                f"detection failed: {type(exc).__name__}: {exc}",
                throttle_duration_sec=5.0,
            )
        finally:
            self._busy.release()

    def _publish(self, msg: Image, detections, frame: np.ndarray) -> None:
        target = None
        if self._target_class:
            target = next(
                (d for d in detections if d.class_name == self._target_class), None
            )
        if target is None:
            target = detections[0] if detections else None

        if self._detection_pub is not None:
            self._detection_pub.publish(self._build_array_msg(msg, detections))
        elif self._json_pub is not None:
            payload = {
                "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
                "frame_id": msg.header.frame_id,
                "width": int(msg.width),
                "height": int(msg.height),
                "detections": [
                    {
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "score": round(d.confidence, 4),
                        "xyxy": [round(v, 2) for v in d.xyxy],
                        "center": [round(v, 2) for v in d.center],
                    }
                    for d in detections
                ],
                "target": None
                if target is None
                else {
                    "class_name": target.class_name,
                    "score": round(target.confidence, 4),
                    "center": [round(v, 2) for v in target.center],
                },
            }
            self._json_pub.publish(String(data=json.dumps(payload)))

        if (
            self._image_pub is not None
            and self._image_pub.get_subscription_count() > 0
        ):
            annotated = draw_detections(
                frame,
                detections,
                crosshair=True,
                max_width=self._annotated_max_width,
            )
            self._image_pub.publish(
                array_to_rgb8_image_msg(
                    annotated, frame_id=msg.header.frame_id, stamp=msg.header.stamp
                )
            )

    def _build_array_msg(self, msg: Image, detections):
        array_msg = self._vision.array()
        array_msg.header = msg.header
        for det in detections:
            detection = self._vision.detection()
            detection.header = msg.header
            detection.id = det.class_name
            detection.results.append(
                self._vision.hypothesis(det.class_id, det.confidence)
            )
            x0, y0, x1, y1 = det.xyxy
            detection.bbox.center.position.x = (x0 + x1) / 2.0
            detection.bbox.center.position.y = (y0 + y1) / 2.0
            detection.bbox.size_x = x1 - x0
            detection.bbox.size_y = y1 - y0
            array_msg.detections.append(detection)
        return array_msg

    def _log_status(
        self,
        detections,
        *,
        convert_ms: float,
        yolo_ms: float,
        publish_ms: float,
    ) -> None:
        now = time.monotonic()
        elapsed = max(now - self._last_log, 1e-6)
        rate = self._log_every / elapsed
        self._last_log = now
        summary = ", ".join(
            f"{d.class_name}@{d.confidence:.2f}" for d in detections[:4]
        )
        self.get_logger().info(
            f"{rate:.1f} Hz, {len(detections)} detections"
            + (f", dropped {self._dropped} frames" if self._dropped else "")
            + (f" [{summary}]" if summary else "")
            + (
                f" timing_ms convert={convert_ms:.1f} yolo={yolo_ms:.1f} "
                f"publish={publish_ms:.1f}"
            )
        )


def main(args=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.detector_node")

    rclpy.init(args=args)
    node = None
    try:
        node = DetectorNode()
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
