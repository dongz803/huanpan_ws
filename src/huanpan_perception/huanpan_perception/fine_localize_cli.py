"""Offline YOLO ROI -> refined corners -> planar pose evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from .dataset import list_images
from .detector import RackDetector
from .fine_localization import (
    CameraModel,
    coarse_rectangle,
    fit_depth_plane,
    refine_rectangle_candidates,
    select_best_planar_pose,
    solve_planar_pose,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--camera-info", required=True)
    parser.add_argument("--depth-dir", default="")
    parser.add_argument("--output", default="outputs/perception/fine_localization")
    parser.add_argument("--mode", choices=("edges", "bbox"), default="edges")
    parser.add_argument("--target-class", default="slot_opening")
    parser.add_argument("--slot-width-mm", type=float, default=100.0)
    parser.add_argument("--slot-height-mm", type=float, default=25.0)
    parser.add_argument("--min-refinement-quality", type=float, default=0.18)
    parser.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args(argv)


def _angle_deviation_deg(quaternions: np.ndarray) -> np.ndarray:
    aligned = quaternions.copy()
    for index in range(1, len(aligned)):
        if np.dot(aligned[0], aligned[index]) < 0:
            aligned[index] *= -1
    mean = np.mean(aligned, axis=0)
    mean /= max(np.linalg.norm(mean), 1e-12)
    dots = np.clip(np.abs(aligned @ mean), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _annotate(image_rgb, bbox, corners, record):
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if bbox is not None:
        x0, y0, x1, y1 = np.rint(bbox).astype(int)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 210, 255), 2)
    if corners is not None:
        polygon = np.rint(corners).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        for index, point in enumerate(polygon):
            cv2.circle(canvas, tuple(point), 4, (255, 70, 20), -1)
            cv2.putText(
                canvas,
                str(index),
                tuple(point + (4, -4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    text = record.get("reason", "")
    if record.get("valid"):
        xyz = np.asarray(record["translation_m"]) * 1000.0
        text = "XYZ mm %.1f %.1f %.1f  err %.2fpx" % (
            *xyz,
            record["reprojection_error_px"],
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text[:100],
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas


def run(args) -> dict:
    camera_data = yaml.safe_load(Path(args.camera_info).read_text(encoding="utf-8"))
    camera = CameraModel.from_mapping(camera_data)
    detector = RackDetector(
        args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device="cpu",
    )
    images = list_images(Path(args.images))
    if args.max_images > 0:
        images = images[: args.max_images]
    output = Path(args.output)
    annotated_dir = output / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None

    records = []
    for image_path in images:
        frame = np.asarray(Image.open(image_path).convert("RGB"))
        bbox = None
        corners = None
        record = {"image": str(image_path), "valid": False}
        try:
            detections = detector.detect(frame)
            targets = [d for d in detections if d.class_name == args.target_class]
            target = detector.largest(targets or detections)
            if target is None:
                raise ValueError("no detection")
            bbox = target.xyxy
            if args.mode == "edges":
                rectangles = refine_rectangle_candidates(frame, bbox)
                rectangles = [
                    candidate
                    for candidate in rectangles
                    if candidate.quality >= args.min_refinement_quality
                ]
                if not rectangles:
                    raise ValueError(
                        "no edge quadrilateral reaches quality %.2f"
                        % args.min_refinement_quality
                    )
                rectangle, pose = select_best_planar_pose(
                    rectangles,
                    camera,
                    width_m=args.slot_width_mm / 1000.0,
                    height_m=args.slot_height_mm / 1000.0,
                )
            else:
                rectangle = coarse_rectangle(bbox)
                pose = solve_planar_pose(
                    rectangle.corners,
                    camera,
                    width_m=args.slot_width_mm / 1000.0,
                    height_m=args.slot_height_mm / 1000.0,
                )
            corners = rectangle.corners
            if pose.reprojection_error_px > args.max_reprojection_error_px:
                raise ValueError(
                    f"reprojection error {pose.reprojection_error_px:.2f}px exceeds "
                    f"{args.max_reprojection_error_px:.2f}px"
                )
            record.update(
                {
                    "valid": True,
                    "coarse_confidence": target.confidence,
                    "bbox_xyxy": list(target.xyxy),
                    "corners": corners.tolist(),
                    "refinement_quality": rectangle.quality,
                    "refinement_method": rectangle.method,
                    "translation_m": pose.translation.tolist(),
                    "quaternion_xyzw": pose.quaternion_xyzw.tolist(),
                    "reprojection_error_px": pose.reprojection_error_px,
                }
            )
            if depth_dir is not None:
                depth_path = depth_dir / image_path.name
                if depth_path.exists():
                    depth = np.asarray(Image.open(depth_path))
                    plane = fit_depth_plane(depth, corners, camera)
                    if plane is not None:
                        record["depth"] = {
                            "median_m": plane.median_depth_m,
                            "valid_ratio": plane.valid_ratio,
                            "plane_distance_m": plane.distance_m,
                            "inlier_ratio": plane.inlier_ratio,
                            "rmse_m": plane.rmse_m,
                        }
        except Exception as exc:  # noqa: BLE001 - every failure belongs in report
            record["reason"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        cv2.imwrite(
            str(annotated_dir / f"{image_path.stem}_fine.png"),
            _annotate(frame, bbox, corners, record),
        )

    valid = [record for record in records if record["valid"]]
    summary = {
        "images": len(records),
        "valid": len(valid),
        "success_rate": (len(valid) / len(records)) if records else 0.0,
        "mode": args.mode,
    }
    if valid:
        translations = np.asarray([record["translation_m"] for record in valid])
        quaternions = np.asarray([record["quaternion_xyzw"] for record in valid])
        reprojection = np.asarray([record["reprojection_error_px"] for record in valid])
        summary.update(
            {
                "translation_mean_m": np.mean(translations, axis=0).tolist(),
                "translation_std_mm": (1000.0 * np.std(translations, axis=0)).tolist(),
                "rotation_deviation_deg_mean": float(np.mean(_angle_deviation_deg(quaternions))),
                "rotation_deviation_deg_max": float(np.max(_angle_deviation_deg(quaternions))),
                "reprojection_error_px_mean": float(np.mean(reprojection)),
                "reprojection_error_px_max": float(np.max(reprojection)),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.fine_localize_cli")
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
