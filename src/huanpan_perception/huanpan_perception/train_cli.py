"""Train the rack/slot detector on a real-image dataset.

Typical use after capturing and labelling::

    ros2 run huanpan_perception train_detector \
        --dataset-root outputs/perception/datasets/rack_grid_v1 \
        --epochs 60 --imgsz 640 --batch 8

The command prepares the train/val split from ``<root>/images`` and
``<root>/labels``, writes ``<root>/data/data.yaml`` and then fine-tunes
``yolov8n`` on the CPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .dataset import build_split
from .detector import DEFAULT_CLASS_NAMES


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="outputs/perception/datasets/rack_grid_v1",
        help="root holding images/ and labels/ (default: %(default)s)",
    )
    parser.add_argument("--images", default="", help="override the images directory")
    parser.add_argument("--labels", default="", help="override the labels directory")
    parser.add_argument("--data", default="", help="use an existing data.yaml instead")
    parser.add_argument("--class-names", default="server_unit", help="comma separated")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-prepare", action="store_true", help="skip split building")
    parser.add_argument(
        "--group-split",
        default="auto",
        choices=["auto", "group", "random"],
        help="how frames are assigned to the validation split; \"auto\" groups "
        "by clip prefix when there is more than one, which prevents "
        "near-identical video frames leaking across the split",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="outputs/perception/runs")
    parser.add_argument("--name", default="server_unit")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args(argv)


def configure_ultralytics(project: Path) -> None:
    """Keep Ultralytics settings and runs inside the workspace."""
    config_dir = project.parent / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def resolve_model(model: str) -> str:
    """Map a bare model name to a file under the workspace weights directory.

    Ultralytics downloads a bare name such as ``yolov8n.pt`` into the *current
    working directory*, which litters the workspace root. Bare names are sent to
    ``outputs/perception/weights/`` instead.
    """
    candidate = Path(model).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate)
    weights_dir = Path("outputs") / "perception" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return str(weights_dir / candidate.name)


def main(argv=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.train_cli")

    args = parse_args(argv)
    # Ultralytics resolves a relative `project` against its own `runs_dir`
    # setting, which silently nests the output directory. Always pass absolute.
    project = Path(args.project).expanduser().resolve()
    configure_ultralytics(project)

    from ultralytics import YOLO, settings

    settings.update({"runs_dir": str(project)})

    class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    data_yaml = Path(args.data).expanduser() if args.data else None

    if data_yaml is None and not args.no_prepare:
        root = Path(args.dataset_root).expanduser()
        images_dir = Path(args.images).expanduser() if args.images else root / "images"
        labels_dir = Path(args.labels).expanduser() if args.labels else root / "labels"
        stats = build_split(
            images_dir,
            labels_dir,
            root / "data",
            class_names=class_names or list(DEFAULT_CLASS_NAMES),
            val_ratio=args.val_ratio,
            seed=args.seed,
            group_by=args.group_split,
        )
        data_yaml = stats.data_yaml
        print(
            f"[train_detector] split: train={stats.train} val={stats.val} "
            f"unlabeled_skipped={stats.unlabeled} classes={stats.class_counts}"
        )
        if stats.split_mode == "group":
            groups = stats.groups or {}
            print(
                f"[train_detector] split by source group ({len(groups)} group(s): "
                + ", ".join(f"{name}x{count}" for name, count in sorted(groups.items()))
                + ") so near-identical frames from one clip stay on one side"
            )
        else:
            largest = max((stats.groups or {"": 0}).values(), default=0)
            if largest > 1:
                print(
                    "[train_detector] WARNING: random split, but frames sharing a "
                    f"name prefix were found (largest group = {largest}). If they "
                    "came from one video, neighbouring frames are near-copies and "
                    "the validation mAP will be optimistically wrong. Record "
                    "several clips and name them <clip>_00001.png."
                )
            else:
                print("[train_detector] random split (frames look independently named).")
        if stats.train < 10:
            print(
                "[train_detector] warning: fewer than 10 training images; "
                "expect an unstable model"
            )
    elif data_yaml is None:
        data_yaml = Path(args.dataset_root).expanduser() / "data" / "data.yaml"

    if not data_yaml.is_file():
        print(f"[train_detector] data.yaml not found: {data_yaml}")
        return 2

    model = YOLO(resolve_model(args.model))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
        deterministic=True,
        verbose=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", project / args.name))
    best = save_dir / "weights" / "best.pt"
    result = {"data": str(data_yaml), "save_dir": str(save_dir), "weights": str(best)}
    if args.export_onnx and best.is_file():
        exported = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, device=args.device)
        result["onnx"] = str(exported)
    (save_dir / "train_invocation.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
