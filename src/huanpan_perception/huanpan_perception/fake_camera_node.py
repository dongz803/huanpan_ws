"""Publish images from disk as a fake RealSense colour stream.

Lets the whole perception chain be tested before (or without) a working
``realsense2_camera`` install::

    ros2 run huanpan_perception fake_camera_node --ros-args \
        -p source:=outputs/perception/datasets/rack_grid_v1/images \
        -p image_topic:=/camera/camera/color/image_raw -p rate_hz:=5.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_PARAMETERS = {
    "source": "",
    "image_topic": "/camera/camera/color/image_raw",
    "rate_hz": 5.0,
    "loop": True,
    "frame_id": "fake_color_optical_frame",
}


class FakeCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("huanpan_fake_camera")
        for name, default in DEFAULT_PARAMETERS.items():
            self.declare_parameter(name, default)

        source = Path(self.get_parameter("source").value).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"source not found: {source}")
        if source.is_dir():
            self._images = sorted(
                path
                for path in source.rglob("*")
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            self._images = [source]
        if not self._images:
            raise FileNotFoundError(f"no images under {source}")

        self._loop = bool(self.get_parameter("loop").value)
        self._frame_id = self.get_parameter("frame_id").value
        self._index = 0
        self._cached: dict[Path, np.ndarray] = {}

        rate = max(float(self.get_parameter("rate_hz").value), 0.1)
        self._publisher = self.create_publisher(
            ImageMsg, self.get_parameter("image_topic").value, qos_profile_sensor_data
        )
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"fake camera: {len(self._images)} images at {rate:.1f} Hz on "
            f"{self.get_parameter('image_topic').value}"
        )

    def _frame(self, path: Path) -> np.ndarray:
        if path not in self._cached:
            self._cached[path] = np.asarray(Image.open(path).convert("RGB"))
        return self._cached[path]

    def _tick(self) -> None:
        if self._index >= len(self._images):
            if not self._loop:
                return
            self._index = 0
        path = self._images[self._index]
        self._index += 1

        array = self._frame(path)
        msg = ImageMsg()
        msg.height, msg.width = int(array.shape[0]), int(array.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(array.shape[1]) * 3
        msg.data = np.ascontiguousarray(array).tobytes()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(msg)


def main(args=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.fake_camera_node")

    rclpy.init(args=args)
    node = None
    try:
        node = FakeCameraNode()
        rclpy.spin(node)
    except (FileNotFoundError, KeyError) as exc:
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
