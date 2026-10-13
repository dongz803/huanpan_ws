"""Label formatting, dataset metadata and overlay drawing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .scene import Detection


def to_yolo_lines(
    detections: Sequence[Detection], width: int, height: int
) -> list[str]:
    """Ultralytics YOLO detection lines: ``class cx cy w h`` normalised."""
    lines = []
    for det in detections:
        x0, y0, x1, y1 = det.xyxy
        cx = (x0 + x1) / 2.0 / width
        cy = (y0 + y1) / 2.0 / height
        w = (x1 - x0) / width
        h = (y1 - y0) / height
        lines.append(
            f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )
    return lines


def write_labels(path: Path, detections: Sequence[Detection], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(to_yolo_lines(detections, width, height)) + "\n", encoding="utf-8")


def detection_to_dict(det: Detection) -> dict:
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "slot": det.slot,
        "xyxy": [round(float(v), 3) for v in det.xyxy],
        "center_px": [round(float(v), 3) for v in det.center_px],
        "depth_m": round(float(det.depth_m), 5),
        "entry_position": [round(float(v), 6) for v in det.entry_position],
        "entry_quat_wxyz": [round(float(v), 6) for v in det.entry_quat_wxyz],
    }


def write_meta(
    path: Path,
    *,
    intrinsics: dict,
    class_names: Sequence[str],
    samples: Iterable[dict],
) -> None:
    payload = {
        "intrinsics": intrinsics,
        "class_names": list(class_names),
        "samples": list(samples),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_data_yaml(
    path: Path,
    dataset_root: Path,
    class_names: Sequence[str],
    splits: Sequence[str],
) -> None:
    """Write an Ultralytics ``data.yaml``.

    Ultralytics resolves relative dataset paths against its own ``datasets_dir``
    setting, so absolute paths are used here.
    """
    root = dataset_root.resolve()
    lines = [
        f"path: {root}",
        f"train: images/{splits[0]}",
        f"val: images/{splits[1]}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


PALETTE = {
    0: (80, 220, 120),  # slot
    1: (255, 170, 60),  # slot_grid
}


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    *,
    labels: bool = True,
) -> Image.Image:
    """Draw ground-truth boxes on an RGB uint8 array."""
    canvas = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        color = PALETTE.get(det.class_id, (255, 80, 80))
        x0, y0, x1, y1 = det.xyxy
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if labels:
            text = det.class_name if det.slot < 0 else f"{det.class_name}:{det.slot:02d}"
            draw.text((x0 + 3, y0 + 2), text, fill=color)
    return canvas


def draw_predictions(
    image: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    *,
    matched_slots: Sequence[int] | None = None,
) -> Image.Image:
    """Draw predicted boxes on an RGB uint8 array."""
    canvas = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for index, (box, class_id, conf) in enumerate(
        zip(boxes_xyxy, class_ids, confidences)
    ):
        color = PALETTE.get(int(class_id), (255, 80, 80))
        x0, y0, x1, y1 = (float(v) for v in box)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        text = f"{conf:.2f}"
        if matched_slots is not None and index < len(matched_slots):
            slot = matched_slots[index]
            text = f"{text} slot{slot:02d}" if slot >= 0 else text
        draw.text((x0 + 3, y0 + 2), text, fill=color)
    return canvas


def montage(images: Sequence[Image.Image], columns: int = 3, pad: int = 4) -> Image.Image:
    if not images:
        raise ValueError("montage needs at least one image")
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * width + (columns + 1) * pad, rows * height + (rows + 1) * pad),
        (18, 24, 32),
    )
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        canvas.paste(image, (pad + col * (width + pad), pad + row * (height + pad)))
    return canvas
