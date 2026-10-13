"""Evaluate a trained detector and report task-relevant localisation errors.

``mAP`` tells you whether the box is right; for the insertion hand-off the
number that matters is how far the predicted rack/slot centre sits from the
true centre. With a known camera intrinsic (``fx``) that pixel error converts
directly to millimetres at the measured depth.

Usage::

    ros2 run huanpan_perception evaluate_detector \
        --model outputs/perception/runs/server_unit/weights/best.pt \
        --dataset-root outputs/perception/datasets/rack_grid_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import list_images
from .detector import RackDetector
from .viz import draw_detections


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset-root",
        default="outputs/perception/datasets/rack_grid_v1",
        help="root that contains data/ and labels/",
    )
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--fx", type=float, default=0.0, help="camera fx in px, for mm errors")
    parser.add_argument("--depth-m", type=float, default=0.0, help="nominal slot depth, for mm errors")
    parser.add_argument("--output", default="outputs/perception/eval")
    parser.add_argument("--max-figures", type=int, default=6)
    return parser.parse_args(argv)


def read_gt_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if not label_path.is_file():
        return np.zeros((0, 4))
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = (float(v) for v in parts)
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
    return np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(ious: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    pairs = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for gt_index, pred_index in np.dstack(
        np.unravel_index(np.argsort(-ious, axis=None), ious.shape)
    )[0]:
        if ious[gt_index, pred_index] < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(int(gt_index))
        used_pred.add(int(pred_index))
        pairs.append((int(gt_index), int(pred_index), float(ious[gt_index, pred_index])))
    return pairs


def main(argv=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.eval_cli")

    args = parse_args(argv)
    root = Path(args.dataset_root).expanduser()
    images_dir = root / "data" / "images" / args.split
    labels_dir = root / "data" / "labels" / args.split
    if not images_dir.is_dir():
        print(f"[evaluate_detector] no split images at {images_dir}; run train_detector first")
        return 2

    images = list_images(images_dir)
    if not images:
        print(f"[evaluate_detector] no images in {images_dir}")
        return 2

    detector = RackDetector(
        args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device
    ).load()

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_ious: list[float] = []
    center_errors_px: list[float] = []
    gt_total = 0
    matched_total = 0
    predicted_total = 0
    figures: list[Image.Image] = []

    for image_path in images:
        frame = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = frame.shape[:2]
        gt_boxes = read_gt_boxes(labels_dir / f"{image_path.stem}.txt", width, height)
        detections = detector.detect(frame)
        pred_boxes = np.array([d.xyxy for d in detections], dtype=float).reshape(-1, 4)
        gt_total += len(gt_boxes)
        predicted_total += len(pred_boxes)

        pairs = greedy_match(iou_matrix(gt_boxes, pred_boxes), args.match_iou)
        matched_total += len(pairs)
        for gt_index, pred_index, iou in pairs:
            matched_ious.append(iou)
            gt_center = np.array(
                [
                    (gt_boxes[gt_index][0] + gt_boxes[gt_index][2]) / 2.0,
                    (gt_boxes[gt_index][1] + gt_boxes[gt_index][3]) / 2.0,
                ]
            )
            center_errors_px.append(
                float(np.linalg.norm(np.array(detections[pred_index].center) - gt_center))
            )
        if len(figures) < args.max_figures:
            figures.append(
                Image.fromarray(draw_detections(frame, detections, crosshair=True))
            )

    center_errors_px_arr = np.array(center_errors_px) if center_errors_px else np.zeros(0)
    metrics = {
        "model": args.model,
        "split": args.split,
        "images": len(images),
        "gt_instances": gt_total,
        "matched_instances": matched_total,
        "predicted_instances": predicted_total,
        "detection_rate": (matched_total / gt_total) if gt_total else 0.0,
        "precision_at_match_iou": (matched_total / predicted_total)
        if predicted_total
        else 0.0,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_center_error_px": float(center_errors_px_arr.mean())
        if center_errors_px_arr.size
        else None,
        "p95_center_error_px": float(np.percentile(center_errors_px_arr, 95))
        if center_errors_px_arr.size
        else None,
    }
    if args.fx > 0 and args.depth_m > 0 and center_errors_px_arr.size:
        mm_per_px = args.depth_m * 1000.0 / args.fx
        metrics["mm_per_px_at_nominal_depth"] = mm_per_px
        metrics["mean_center_error_mm"] = float(center_errors_px_arr.mean() * mm_per_px)
        metrics["p95_center_error_mm"] = float(
            np.percentile(center_errors_px_arr, 95) * mm_per_px
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if figures:
        columns = min(3, len(figures))
        rows = (len(figures) + columns - 1) // columns
        width = max(figure.width for figure in figures)
        height = max(figure.height for figure in figures)
        canvas = Image.new(
            "RGB", (columns * width + 4 * (columns + 1), rows * height + 4 * (rows + 1)),
            (18, 24, 32),
        )
        for index, figure in enumerate(figures):
            row, col = divmod(index, columns)
            canvas.paste(figure, (4 + col * (width + 4), 4 + row * (height + 4)))
        canvas.save(output_dir / "predictions.png")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
