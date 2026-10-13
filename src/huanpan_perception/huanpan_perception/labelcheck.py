"""Label consistency checks for a hand-annotated detection dataset.

At ten images a wrong box is obvious; at three hundred it is not. This module
looks for the mistakes that actually happen when one person labels a few hundred
photos of the same object:

* an image with no label, or an empty label file;
* a different number of boxes than the rest of the dataset;
* a box whose aspect ratio or area is far from the dataset median (the usual
  signature of a mis-drawn or mis-dragged box);
* a box that covers almost the whole frame, or is vanishingly small.

It is deliberately statistics-only: no model, no training, so it runs in
milliseconds and can be used before every training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from PIL import Image

from .dataset import label_path_for, list_images


@dataclass
class BoxRecord:
    image: Path
    index: int
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    image_width: int
    image_height: int

    @property
    def pixel_width(self) -> float:
        return self.w * self.image_width

    @property
    def pixel_height(self) -> float:
        return self.h * self.image_height

    @property
    def aspect(self) -> float:
        height = self.pixel_height
        return self.pixel_width / height if height > 1e-9 else float("inf")

    @property
    def area_fraction(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)


@dataclass
class ImageReport:
    image: Path
    box_count: int
    issues: list[str] = field(default_factory=list)
    severity: float = 0.0


@dataclass
class CheckReport:
    images: int
    labelled: int
    unlabelled: list[Path]
    empty_labels: list[Path]
    box_count_histogram: dict[int, int]
    median_aspect: float
    median_area_fraction: float
    flagged: list[ImageReport]


def load_boxes(labels_dir: Path, image_path: Path) -> list[BoxRecord]:
    """Parse one YOLO label file against its image size."""
    label = label_path_for(labels_dir, image_path)
    if not label.is_file():
        return []
    with Image.open(image_path) as handle:
        width, height = handle.size

    boxes: list[BoxRecord] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        boxes.append(
            BoxRecord(
                image=image_path,
                index=len(boxes),
                class_id=class_id,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                image_width=width,
                image_height=height,
            )
        )
    return boxes


def check_dataset(
    images_dir: Path,
    labels_dir: Path,
    *,
    aspect_tolerance: float = 2.0,
    area_tolerance: float = 2.5,
    tiny_area: float = 0.005,
    huge_area: float = 0.9,
) -> CheckReport:
    """Compare every label against the dataset-wide typical box."""
    images = list_images(images_dir)
    per_image_boxes = {image: load_boxes(labels_dir, image) for image in images}

    all_boxes = [box for boxes in per_image_boxes.values() for box in boxes]
    aspects = [box.aspect for box in all_boxes if box.aspect != float("inf")]
    areas = [box.area_fraction for box in all_boxes]
    median_aspect = median(aspects) if aspects else 0.0
    median_area = median(areas) if areas else 0.0

    histogram: dict[int, int] = {}
    for boxes in per_image_boxes.values():
        histogram[len(boxes)] = histogram.get(len(boxes), 0) + 1
    typical_count = max(histogram, key=lambda key: histogram[key]) if histogram else 1

    unlabelled: list[Path] = []
    empty_labels: list[Path] = []
    flagged: list[ImageReport] = []

    for image, boxes in per_image_boxes.items():
        label = label_path_for(labels_dir, image)
        report = ImageReport(image=image, box_count=len(boxes))

        if not label.is_file():
            unlabelled.append(image)
            report.issues.append("没有标签文件")
            report.severity = 10.0
            flagged.append(report)
            continue
        if not boxes:
            empty_labels.append(image)
            report.issues.append("标签文件是空的")
            report.severity = 9.0
            flagged.append(report)
            continue

        if len(boxes) != typical_count:
            report.issues.append(
                f"框数量 {len(boxes)} 与数据集常见值 {typical_count} 不同"
            )
            report.severity += 3.0

        for box in boxes:
            if median_aspect > 0:
                ratio = box.aspect / median_aspect
                if ratio > aspect_tolerance or ratio < 1 / aspect_tolerance:
                    report.issues.append(
                        f"第 {box.index + 1} 个框宽高比 {box.aspect:.2f}，"
                        f"中位数 {median_aspect:.2f}（{ratio:.2f}x）"
                    )
                    report.severity += min(abs(ratio - 1.0) * 2.0, 6.0)
            if median_area > 0:
                ratio = box.area_fraction / median_area
                if ratio > area_tolerance or ratio < 1 / area_tolerance:
                    report.issues.append(
                        f"第 {box.index + 1} 个框面积占比 {box.area_fraction:.3f}，"
                        f"中位数 {median_area:.3f}（{ratio:.2f}x）"
                    )
                    report.severity += min(abs(ratio - 1.0) * 2.0, 6.0)
            if box.area_fraction < tiny_area:
                report.issues.append(f"第 {box.index + 1} 个框过小")
                report.severity += 4.0
            if box.area_fraction > huge_area and len(boxes) == 1:
                report.issues.append(f"第 {box.index + 1} 个框几乎覆盖整张图")
                report.severity += 2.0
            if not (0.0 <= box.cx <= 1.0 and 0.0 <= box.cy <= 1.0):
                report.issues.append(f"第 {box.index + 1} 个框中心越界")
                report.severity += 8.0

        if report.issues:
            flagged.append(report)

    flagged.sort(key=lambda item: item.severity, reverse=True)
    return CheckReport(
        images=len(images),
        labelled=len(all_boxes),
        unlabelled=unlabelled,
        empty_labels=empty_labels,
        box_count_histogram=histogram,
        median_aspect=median_aspect,
        median_area_fraction=median_area,
        flagged=flagged,
    )


def format_report(report: CheckReport, max_listed: int = 25) -> str:
    lines = [
        f"图片总数        : {report.images}",
        f"标注框总数      : {report.labelled}",
        f"每个图的框数量  : "
        + ", ".join(
            f"{count} 框 x {times}"
            for count, times in sorted(report.box_count_histogram.items())
        ),
        f"宽高比中位数    : {report.median_aspect:.2f}",
        f"面积占比中位数  : {report.median_area_fraction:.3f}",
        f"未标注          : {len(report.unlabelled)} 张",
        f"空标签          : {len(report.empty_labels)} 张",
        f"可疑            : {len(report.flagged)} 张",
    ]
    if report.unlabelled:
        lines.append("  未标注: " + ", ".join(p.name for p in report.unlabelled[:max_listed]))
    if report.empty_labels:
        lines.append("  空标签: " + ", ".join(p.name for p in report.empty_labels[:max_listed]))
    if report.flagged:
        lines.append("")
        lines.append("按可疑程度排序（先看最上面的）:")
        for item in report.flagged[:max_listed]:
            lines.append(f"  [{item.severity:5.1f}] {item.image.name}")
            for issue in item.issues:
                lines.append(f"          - {issue}")
    return "\n".join(lines)
