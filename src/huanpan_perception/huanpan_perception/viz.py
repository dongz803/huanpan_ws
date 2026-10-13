"""Drawing helpers for detection overlays (Pillow only, no OpenCV needed)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .detector import Detection

CLASS_COLORS = {
    0: (90, 220, 130),
    1: (255, 170, 60),
    2: (120, 180, 255),
}
FALLBACK_COLOR = (255, 90, 90)


def color_for(class_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS.get(int(class_id), FALLBACK_COLOR)


def draw_detections(
    image_rgb: np.ndarray,
    detections: Sequence[Detection],
    *,
    show_labels: bool = True,
    crosshair: bool = False,
    color: tuple[int, int, int] | None = None,
    width: int = 2,
    max_width: int = 0,
) -> np.ndarray:
    """Return a copy of the RGB image with boxes drawn on it.

    ``color`` overrides the per-class colour, which is what the ground-truth
    comparison uses so predictions stand out against green GT boxes.
    """
    coordinate_scale = 1.0
    preview = np.ascontiguousarray(image_rgb)
    if max_width > 0 and preview.shape[1] > max_width:
        import cv2

        coordinate_scale = max_width / float(preview.shape[1])
        preview = cv2.resize(
            preview,
            (max_width, max(1, int(round(preview.shape[0] * coordinate_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    canvas = Image.fromarray(preview).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        box_color = color or color_for(det.class_id)
        x0, y0, x1, y1 = (
            value * coordinate_scale for value in det.xyxy
        )
        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=width)
        if show_labels:
            draw.text(
                (x0 + 3, y0 + 2),
                f"{det.class_name} {det.confidence:.2f}",
                fill=box_color,
            )
        if crosshair:
            cx, cy = det.center
            cx *= coordinate_scale
            cy *= coordinate_scale
            draw.line([cx - 8, cy, cx + 8, cy], fill=box_color, width=width)
            draw.line([cx, cy - 8, cx, cy + 8], fill=box_color, width=width)
    return np.asarray(canvas)


def montage(
    images: Sequence[np.ndarray], columns: int = 3, pad: int = 4
) -> np.ndarray:
    """Tile RGB arrays into one padded canvas."""
    if not images:
        raise ValueError("montage needs at least one image")
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = np.full(
        (rows * height + (rows + 1) * pad, columns * width + (columns + 1) * pad, 3),
        22,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        y = pad + row * (height + pad)
        x = pad + col * (width + pad)
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    return canvas


def draw_label_boxes(
    image_rgb: np.ndarray, label_path, class_name: str = "server_unit"
) -> np.ndarray:
    """Draw YOLO-format boxes from ``label_path`` onto an RGB image.

    Used to sanity-check hand-made labels before spending time on training.
    """
    from pathlib import Path

    canvas = Image.fromarray(np.ascontiguousarray(image_rgb)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    path = Path(label_path)
    if path.is_file():
        height, width = image_rgb.shape[:2]
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = (float(v) for v in parts)
            color = color_for(int(float(parts[0])))
            draw.rectangle(
                [
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ],
                outline=color,
                width=max(2, width // 300),
            )
            draw.text(
                ((cx - w / 2) * width + 4, (cy - h / 2) * height + 3),
                class_name,
                fill=color,
            )
    else:
        draw.text((10, 10), "NO LABEL", fill=(255, 90, 90))
    return np.asarray(canvas)
