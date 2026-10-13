"""ROS-free YOLO detector wrapper.

Keeping the detector free of any ROS import makes it unit-testable and lets the
offline CLI, the capture tool and the ROS node share exactly one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .image_utils import to_bgr

DEFAULT_CLASS_NAMES = ("server_unit",)


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


class DetectorError(RuntimeError):
    """Raised when the detector cannot be constructed or run."""


class RackDetector:
    """Thin, predictable wrapper around an Ultralytics YOLO model.

    The first model detects a single class, ``server_unit``: the server chassis
    that the robot inserts drives into (the unit with its own front faceplate),
    not the surrounding loose drive caddies. ``largest`` is the convenience
    accessor the runtime uses to publish one coarse localisation result.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu",
        max_det: int = 20,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.max_det = int(max_det)
        self.class_names = tuple(class_names) if class_names else DEFAULT_CLASS_NAMES
        self._model = None

    # ------------------------------------------------------------- lifecycle
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "RackDetector":
        """Import Ultralytics and load the weights.

        The import is deferred because it pulls in Torch, which is by far the
        slowest part of node start-up and is irrelevant to the pure-python
        tests and tooling.
        """
        if self._model is not None:
            return self
        if not self.model_path.exists():
            raise DetectorError(
                f"model weights not found: {self.model_path}\n"
                "Train a detector first, for example:\n"
                "  ros2 run huanpan_perception train_detector "
                "--data <dataset>/data.yaml --epochs 60\n"
                "then point model_path at runs/detect/*/weights/best.pt"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DetectorError(
                "Ultralytics is not importable in this interpreter. Run the "
                "perception nodes with the workspace virtualenv, e.g.\n"
                "  source scripts/perception_env.sh\n"
                "  .venv-perception/bin/python -m huanpan_perception.detector_node"
            ) from exc

        self._model = YOLO(str(self.model_path))
        names = getattr(self._model, "names", None)
        if isinstance(names, dict) and names:
            self.class_names = tuple(str(names[key]) for key in sorted(names))
        return self

    # -------------------------------------------------------------- inference
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Run detection on an RGB ``uint8`` image."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"expected an HxWx3 RGB image, got shape {image_rgb.shape}"
            )
        self.load()
        results = self._model.predict(
            source=to_bgr(image_rgb),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        detections = []
        for box, class_id, confidence in zip(xyxy, classes, confidences):
            name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=name,
                    confidence=float(confidence),
                    xyxy=tuple(float(v) for v in box),
                )
            )
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def largest(self, detections: Sequence[Detection]) -> Detection | None:
        """Highest-confidence detection, or the largest box if scores tie."""
        if not detections:
            return None
        return max(detections, key=lambda det: (det.confidence, det.area))
