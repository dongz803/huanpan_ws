"""Evaluate the trained slot detector and report task-relevant errors.

``mAP`` says how good the boxes are; the number that actually matters for the
insertion pipeline is *how far the predicted slot centre is from the real slot
centre*, in millimetres at the slot depth. This script reports both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import montage, draw_predictions

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--split", default="val", choices=["val", "hard"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-figures", type=int, default=6)
    return parser.parse_args(argv)


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


def match(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, threshold: float
) -> list[tuple[int, int, float]]:
    """Greedy highest-IoU matching; returns ``(gt_index, pred_index, iou)``."""
    ious = iou_matrix(gt_boxes, pred_boxes)
    pairs: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    for gt_index, pred_index in order:
        if ious[gt_index, pred_index] < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(int(gt_index))
        used_pred.add(int(pred_index))
        pairs.append((int(gt_index), int(pred_index), float(ious[gt_index, pred_index])))
    return pairs


def main(argv=None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO

    data_root = args.data if args.data.is_absolute() else WORKSPACE_ROOT / args.data
    meta = json.loads((data_root / "meta.json").read_text(encoding="utf-8"))
    intrinsics = meta["intrinsics"]
    fx = intrinsics["fx"]

    samples = [s for s in meta["samples"] if s["split"] == args.split]
    if not samples:
        raise SystemExit(f"no samples for split {args.split!r} in {data_root}")

    out_dir = args.out or (WORKSPACE_ROOT / "outputs" / "perception" / "eval" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics resolves the validation source from data.yaml, so point a
    # temporary yaml at the requested split.
    temp_yaml = out_dir / f"data_{args.split}.yaml"
    temp_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root.resolve()}",
                f"train: images/{args.split}",
                f"val: images/{args.split}",
                f"nc: {len(meta['class_names'])}",
                "names:",
                *[f"  {i}: {n}" for i, n in enumerate(meta["class_names"])],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model = YOLO(str(args.weights))
    val_metrics = model.val(
        data=str(temp_yaml), split="val", imgsz=args.imgsz, device=args.device, verbose=False
    )

    per_class: dict[str, dict] = {}
    for index, name in enumerate(meta["class_names"]):
        try:
            per_class[name] = {
                "precision": float(val_metrics.box.p[index]),
                "recall": float(val_metrics.box.r[index]),
                "map50": float(val_metrics.box.ap50[index]),
                "map50_95": float(val_metrics.box.ap[index]),
            }
        except (IndexError, TypeError):
            per_class[name] = {}

    figures: list[Image.Image] = []
    matched_ious: list[float] = []
    center_errors_px: list[float] = []
    center_errors_mm: list[float] = []
    grid_center_errors_px: list[float] = []
    total_gt_slots = 0
    total_matched = 0
    total_predictions = 0

    for sample in samples:
        image = np.asarray(Image.open(data_root / sample["image"]).convert("RGB"))
        result = model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        pred_boxes = (
            result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
        )
        pred_classes = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes is not None
            else np.zeros(0, dtype=int)
        )
        pred_conf = (
            result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(0)
        )
        total_predictions += len(pred_boxes)

        gt_slots = [d for d in sample["detections"] if d["slot"] >= 0]
        gt_slot_boxes = np.array([d["xyxy"] for d in gt_slots], dtype=float).reshape(-1, 4)
        total_gt_slots += len(gt_slots)

        slot_mask = pred_classes == 0
        slot_boxes = pred_boxes[slot_mask]
        pairs = match(gt_slot_boxes, slot_boxes, args.match_iou)
        total_matched += len(pairs)

        matched_slots = [-1] * len(pred_boxes)
        slot_indices = np.flatnonzero(slot_mask)
        for gt_index, pred_index, iou in pairs:
            matched_slots[int(slot_indices[pred_index])] = gt_slots[gt_index]["slot"]
            matched_ious.append(iou)
            gt_center = np.array(gt_slots[gt_index]["center_px"], dtype=float)
            x0, y0, x1, y1 = slot_boxes[pred_index]
            pred_center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            error_px = float(np.linalg.norm(pred_center - gt_center))
            center_errors_px.append(error_px)
            depth = float(gt_slots[gt_index]["depth_m"])
            center_errors_mm.append(error_px * depth * 1000.0 / fx)

        # Whole-grid box: coarse "where is the rack" error.
        gt_grid = [d for d in sample["detections"] if d["slot"] < 0]
        grid_mask = pred_classes == 1
        if gt_grid and grid_mask.any():
            gt_grid_box = np.array([gt_grid[0]["xyxy"]], dtype=float)
            grid_pairs = match(gt_grid_box, pred_boxes[grid_mask], 0.3)
            if grid_pairs:
                _, pred_index, _ = grid_pairs[0]
                x0, y0, x1, y1 = pred_boxes[grid_mask][pred_index]
                pred_center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
                gt_center = np.array(gt_grid[0]["center_px"], dtype=float)
                grid_center_errors_px.append(float(np.linalg.norm(pred_center - gt_center)))

        if len(figures) < args.max_figures:
            figures.append(
                draw_predictions(
                    image, pred_boxes, pred_classes, pred_conf, matched_slots=matched_slots
                )
            )

    metrics = {
        "split": args.split,
        "weights": str(args.weights),
        "conf_threshold": args.conf,
        "match_iou": args.match_iou,
        "ultralytics": {
            "map50": float(val_metrics.box.map50),
            "map50_95": float(val_metrics.box.map),
            "per_class": per_class,
        },
        "task_metrics": {
            "gt_slots": total_gt_slots,
            "matched_slots": total_matched,
            "detection_rate_at_iou": (
                total_matched / total_gt_slots if total_gt_slots else 0.0
            ),
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "mean_center_error_px": (
                float(np.mean(center_errors_px)) if center_errors_px else None
            ),
            "mean_center_error_mm": (
                float(np.mean(center_errors_mm)) if center_errors_mm else None
            ),
            "p95_center_error_mm": (
                float(np.percentile(center_errors_mm, 95)) if center_errors_mm else None
            ),
            "mean_grid_center_error_px": (
                float(np.mean(grid_center_errors_px)) if grid_center_errors_px else None
            ),
            "predictions": total_predictions,
        },
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if figures:
        montage(figures, columns=3).save(out_dir / "predictions.png")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
