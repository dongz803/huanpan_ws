"""Train the slot detector with Ultralytics YOLO.

There is no GPU in this workspace, so the default configuration targets CPU:
a small model (``yolov8n``), a modest image size and a short schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = WORKSPACE_ROOT / "outputs" / "perception" / "runs"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="weights or model name")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="slot_yolov8n")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO, settings

    settings.update(
        {
            "datasets_dir": str(WORKSPACE_ROOT / "outputs" / "perception" / "datasets"),
            "runs_dir": str(args.project),
        }
    )

    model_path = args.model
    candidate = Path(model_path)
    if not candidate.is_absolute() and (WORKSPACE_ROOT / candidate).exists():
        model_path = str(WORKSPACE_ROOT / candidate)

    model = YOLO(model_path)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        plots=True,
        val=True,
        deterministic=True,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", args.project / args.name))
    summary = {"save_dir": str(save_dir), "weights": str(save_dir / "weights" / "best.pt")}
    metrics_path = save_dir / "results.csv"
    if metrics_path.exists():
        summary["results_csv"] = str(metrics_path)
    (save_dir / "train_invocation.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
