"""ROS 2 node: capture a real-image dataset from the D435.

Two modes:

``interactive`` (default when stdin is a terminal)
    Move the camera by hand and press Enter to save the current frame. This is
    the fastest way to build a first dataset from a handheld D435.

``interval``
    Save a frame every ``interval_s`` seconds, up to ``max_frames``.

Colour frames are written as PNG. When ``save_depth`` is enabled the aligned
depth frame is written next to the colour frame as a 16-bit PNG in millimetres.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMsg

from message_filters import ApproximateTimeSynchronizer, Subscriber

from .capture_metadata import camera_info_dict, next_capture_index, stamp_to_ns
from .image_utils import image_msg_to_array

DEFAULT_PARAMETERS = {
    "image_topic": "/camera/camera/color/image_raw",
    "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic": "/camera/camera/color/camera_info",
    "output_dir": "",
    "prefix": "capture",
    "mode": "auto",  # auto | interactive | interval
    "interval_s": 1.0,
    "max_frames": 300,
    "save_depth": False,
    "min_interval_s": 0.15,
    # Live OpenCV preview with a keyboard trigger; needs save_depth.
    "preview": False,
    "sync_slop_s": 0.03,
    "sync_queue_size": 15,
}


class CaptureNode(Node):
    def __init__(self) -> None:
        super().__init__("huanpan_capture")
        for name, default in DEFAULT_PARAMETERS.items():
            self.declare_parameter(name, default)

        output_dir = self.get_parameter("output_dir").value
        if not output_dir:
            raise ValueError("parameter 'output_dir' must be set")
        self._output_dir = Path(output_dir).expanduser()
        # `output_dir` is the dataset ROOT: colour frames land in <root>/images
        # and depth in <root>/depth. Passing the images directory itself is an
        # easy mistake that silently produces <root>/images/images.
        if self._output_dir.name in {"images", "depth"}:
            self.get_logger().warning(
                f"output_dir ends with '{self._output_dir.name}'; it should be the "
                f"dataset ROOT, so frames will be written to "
                f"{self._output_dir / 'images'}. Pass the parent directory instead."
            )
        self._image_dir = self._output_dir / "images"
        self._depth_dir = self._output_dir / "depth"
        self._metadata_dir = self._output_dir / "metadata"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        if bool(self.get_parameter("save_depth").value):
            self._depth_dir.mkdir(parents=True, exist_ok=True)

        self._prefix = self.get_parameter("prefix").value
        self._max_frames = int(self.get_parameter("max_frames").value)
        self._save_depth = bool(self.get_parameter("save_depth").value)
        self._min_interval = float(self.get_parameter("min_interval_s").value)

        self._preview = bool(self.get_parameter("preview").value)
        self._preview_window = "huanpan capture  [SPACE] save  [d] depth  [q] quit"
        self._show_depth = False

        mode = self.get_parameter("mode").value
        self._interval_s = float(self.get_parameter("interval_s").value)
        # With a preview window the keyboard arrives through OpenCV, so the
        # stdin check is irrelevant and interactive is always possible.
        interactive = (
            self._preview
            or mode == "interactive"
            or (mode == "auto" and self._stdin_is_tty())
        )
        self._interactive = interactive

        if self._preview and not self._save_depth:
            raise ValueError(
                "preview:=true needs save_depth:=true so the depth view and the "
                "saved frames match; pass save_depth:=true"
            )

        self._count = 0
        self._next_index = next_capture_index(
            self._image_dir, self._depth_dir, self._prefix
        )
        self._stop = False
        self._pending = 0
        self._last_save = 0.0
        self._latest_depth = None
        self._latest_sync_delta_ms = 0.0
        self._camera_info_written = False
        self._lock = threading.Lock()

        self._image_topic = self.get_parameter("image_topic").value
        self._preview_started = time.monotonic()
        self._last_frame_time: float | None = None
        self._last_starved_warning = 0.0

        self._image_sub = Subscriber(
            self, ImageMsg, self._image_topic, qos_profile=qos_profile_sensor_data
        )
        self._info_sub = Subscriber(
            self,
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            qos_profile=qos_profile_sensor_data,
        )
        if self._save_depth:
            self._depth_sub = Subscriber(
                self,
                ImageMsg,
                self.get_parameter("depth_topic").value,
                qos_profile=qos_profile_sensor_data,
            )
            subscribers = [self._image_sub, self._depth_sub, self._info_sub]
        else:
            self._depth_sub = None
            subscribers = [self._image_sub, self._info_sub]

        self._synchronizer = ApproximateTimeSynchronizer(
            subscribers,
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_s").value),
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._on_synced)

        if self._preview:
            self.get_logger().info(
                "preview capture: SPACE/s saves a frame, d toggles the depth view, "
                "q quits"
            )
            # Open the window straight away with a placeholder, and keep it
            # alive from a timer. Without this the window only appeared once the
            # first frame arrived, so a camera that never streamed looked like a
            # hung program and the window keyboard could not be used to quit.
            self._open_preview_window()
            self.create_timer(0.5, self._preview_tick)
        elif self._interactive:
            thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            thread.start()
            self.get_logger().info(
                "interactive capture: press Enter to save a frame, 'q' + Enter to stop"
            )
        else:
            self.get_logger().info(
                f"interval capture: one frame every {self._interval_s:.2f}s, "
                f"up to {self._max_frames} frames"
            )
        self.get_logger().info(
            f"saving into {self._image_dir}; next index for prefix "
            f"'{self._prefix}' is {self._next_index:05d}"
        )

    @property
    def finished(self) -> bool:
        """True once capture stopped on its own (max_frames or 'q')."""
        return self._stop

    # ------------------------------------------------------------------ input
    @staticmethod
    def _stdin_is_tty() -> bool:
        import sys

        return bool(getattr(sys.stdin, "isatty", lambda: False)())

    def _keyboard_loop(self) -> None:
        import sys

        while not self._stop:
            line = sys.stdin.readline()
            if line == "":
                break
            if line.strip().lower() in {"q", "quit", "exit"}:
                self._stop = True
                self.get_logger().info("stopping capture")
                return
            with self._lock:
                self._pending += 1

    def _write_camera_info(self, msg: CameraInfo) -> None:
        if self._camera_info_written:
            return
        import yaml

        path = self._metadata_dir / f"{self._prefix}_camera_info.yaml"
        payload = camera_info_dict(msg)
        payload["image_topic"] = self._image_topic
        payload["depth_topic"] = self.get_parameter("depth_topic").value
        payload["camera_info_topic"] = self.get_parameter("camera_info_topic").value
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._camera_info_written = True

    def _open_preview_window(self) -> None:
        import cv2

        cv2.namedWindow(self._preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._preview_window, 960, 600)

    def _handle_preview_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self._stop = True
        elif key in (ord("s"), ord(" ")):
            if self._last_frame_time is None:
                self.get_logger().warning("no frame yet; nothing to save")
            else:
                with self._lock:
                    self._pending += 1
        elif key == ord("d"):
            self._show_depth = not self._show_depth
            self.get_logger().info(
                f"depth preview {'on' if self._show_depth else 'off'}"
            )

    def _preview_tick(self) -> None:
        """Keep the window alive and show why nothing is happening.

        Frames normally repaint the window from ``_on_image``. This timer only
        steps in when frames have stopped arriving, so the user sees a
        "waiting for frames" panel instead of a blank screen, and can still quit.
        """
        if self._stop:
            return
        now = time.monotonic()
        if self._last_frame_time is not None and now - self._last_frame_time < 0.5:
            return

        import cv2

        uptime = now - self._preview_started
        canvas = np.full((400, 980, 3), 28, dtype=np.uint8)

        if self._last_frame_time is None:
            # Never received anything: the camera or the topic is the problem.
            headline = f"WAITING FOR THE FIRST FRAME   ({uptime:.0f}s)"
            detail = [
                f"topic        : {self._image_topic}",
                f"node uptime  : {uptime:.0f}s",
                "",
                "check:  ros2 topic hz " + self._image_topic,
                "check:  rs-enumerate-devices -s",
                "check:  realsense errors in this terminal",
            ]
        else:
            # Frames were arriving and then stopped: the stream dropped.
            gap = now - self._last_frame_time
            headline = f"STREAM STOPPED   (no frame for {gap:.0f}s)"
            detail = [
                f"topic        : {self._image_topic}",
                f"frames saved : {self._count}",
                f"node uptime  : {uptime:.0f}s",
                "",
                "the driver retries every ~6s; if it recovers this panel clears",
                "if it keeps happening:  sudo dmesg | tail -20",
                "look for 'USB disconnect' / 'SET_CUR UVC control'",
            ]

        lines = [headline] + detail + ["", "[q] quit"]
        for index, line in enumerate(lines):
            colour = (80, 200, 255) if index == 0 else (200, 200, 200)
            cv2.putText(
                canvas,
                line,
                (24, 56 + index * 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colour,
                2,
            )
        cv2.imshow(self._preview_window, canvas)
        self._handle_preview_key(cv2.waitKey(1) & 0xFF)

        if self._last_frame_time is not None:
            gap = now - self._last_frame_time
            if gap - self._last_starved_warning > 10.0:
                self._last_starved_warning = gap
                self.get_logger().error(
                    f"stream stopped: no frame on {self._image_topic} for "
                    f"{gap:.0f}s ({self._count} frames saved so far). The driver "
                    "retries every ~6s. If this repeats, run "
                    "`sudo dmesg | tail -20` and look for 'USB disconnect'."
                )

    def _show_preview(self, frame: np.ndarray) -> None:
        """Live preview with a keyboard trigger.

        Keyboard input goes through the OpenCV window, not stdin, so this also
        works when the node is started by ``ros2 launch`` (where stdin is not a
        terminal).
        """
        import cv2

        display = frame
        if self._show_depth and self._latest_depth is not None:
            depth = self._latest_depth
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)
            valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 3.0)
            span = max(float(depth_m[valid].max()) if valid.any() else 1.0, 0.3)
            scaled = np.clip(255.0 * (1.0 - depth_m / span), 0, 255).astype(np.uint8)
            coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            coloured[~valid] = 0
            if coloured.shape[:2] != display.shape[:2]:
                coloured = cv2.resize(
                    coloured, (display.shape[1], display.shape[0])
                )
            display = cv2.addWeighted(display, 0.55, coloured, 0.45, 0.0)

        if display.ndim == 2:
            display = np.repeat(display[:, :, None], 3, axis=2)
        overlay = display[:, :, ::-1].copy()
        text = (
            f"saved {self._count}/{self._max_frames}   "
            "[SPACE/s] save   [d] depth view   [q] quit"
        )
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            overlay, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.imshow(self._preview_window, overlay)
        self._handle_preview_key(cv2.waitKey(1) & 0xFF)

    def _on_synced(self, color_msg: ImageMsg, *messages) -> None:
        if self._stop or self._count >= self._max_frames:
            return

        if self._save_depth:
            depth_msg, info_msg = messages
            depth = image_msg_to_array(depth_msg)
            stamps = [
                stamp_to_ns(color_msg.header.stamp),
                stamp_to_ns(depth_msg.header.stamp),
                stamp_to_ns(info_msg.header.stamp),
            ]
        else:
            (info_msg,) = messages
            depth_msg = None
            depth = None
            stamps = [
                stamp_to_ns(color_msg.header.stamp),
                stamp_to_ns(info_msg.header.stamp),
            ]

        frame = image_msg_to_array(color_msg)
        if depth is not None and depth.shape[:2] != frame.shape[:2]:
            self.get_logger().error(
                f"rejecting unsized RGB-D pair: color={frame.shape[:2]}, "
                f"depth={depth.shape[:2]}",
                throttle_duration_sec=5.0,
            )
            return
        if (int(info_msg.width), int(info_msg.height)) != (
            int(color_msg.width),
            int(color_msg.height),
        ):
            self.get_logger().error(
                "rejecting pair: CameraInfo dimensions do not match color image",
                throttle_duration_sec=5.0,
            )
            return

        self._latest_depth = depth
        self._latest_sync_delta_ms = (max(stamps) - min(stamps)) / 1_000_000.0
        self._write_camera_info(info_msg)
        now = time.monotonic()
        self._last_frame_time = now

        if self._preview:
            self._show_preview(frame)
            if self._stop:
                return

        if now - self._last_save < self._min_interval:
            return

        if self._interactive:
            with self._lock:
                if self._pending <= 0:
                    return
                self._pending -= 1
        elif now - self._last_save < self._interval_s:
            return

        self._last_save = now
        self._count += 1
        stem = f"{self._prefix}_{self._next_index:05d}"
        self._next_index += 1
        Image.fromarray(frame).save(self._image_dir / f"{stem}.png")

        if self._save_depth:
            if depth.dtype == np.float32:
                depth = np.clip(depth * 1000.0, 0, 65535)
            depth = depth.astype(np.uint16)
            Image.fromarray(depth).save(self._depth_dir / f"{stem}.png")

        valid_depth_ratio = ""
        if depth is not None:
            valid = np.isfinite(depth) & (depth > 0)
            valid_depth_ratio = f"{float(valid.mean()):.6f}"
        manifest = self._output_dir / "frames.csv"
        write_header = not manifest.exists() or manifest.stat().st_size == 0
        with manifest.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(
                    [
                        "stem",
                        "color_stamp_ns",
                        "depth_stamp_ns",
                        "camera_info_stamp_ns",
                        "sync_delta_ms",
                        "width",
                        "height",
                        "valid_depth_ratio",
                    ]
                )
            writer.writerow(
                [
                    stem,
                    stamps[0],
                    "" if depth_msg is None else stamps[1],
                    stamps[-1],
                    f"{self._latest_sync_delta_ms:.3f}",
                    int(color_msg.width),
                    int(color_msg.height),
                    valid_depth_ratio,
                ]
            )

        self.get_logger().info(
            f"saved {stem}.png ({frame.shape[1]}x{frame.shape[0]}) "
            f"sync={self._latest_sync_delta_ms:.1f} ms "
            f"[{self._count}/{self._max_frames}]"
        )
        if self._count >= self._max_frames:
            self.get_logger().info("reached max_frames, stopping capture")
            self._stop = True


def main(args=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.capture_node")

    rclpy.init(args=args)
    node = None
    try:
        node = CaptureNode()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    except (ValueError, KeyError) as exc:
        print(f"[huanpan_perception] {exc}")
        return 2
    finally:
        if node is not None:
            if node._preview:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass
            node.get_logger().info(f"capture finished: {node._count} frames")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
