"""Utility script for preparing the Laser dataset for YOLO training."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LABEL_MAP = {"old": 0, "1": 1, "2": 2, "2+": 3, "3": 4, "3+": 5}
IMAGE_SIZE = 1240
GROUP_SIZE = 8
VAL_RATIO = 1  # number of images per group that go to validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Laser dataset for YOLO training.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/wensheng/jiaqi/ultralytics/yolo_data"),
        help="Directory containing the raw BMP/JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/wensheng/jiaqi/ultralytics/yolo_data/laser"),
        help="Root directory where the YOLO-formatted dataset will be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Width/height of the square BMP images (used for normalization).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing split directories before writing new data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def find_samples(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    bmp_files = sorted(source.glob("*.bmp"))
    if not bmp_files:
        logging.warning("No BMP files were found in %s", source)
    return bmp_files


def ensure_dirs(output: Path, overwrite: bool = False) -> None:
    images_root = output / "images"
    labels_root = output / "labels"
    for root in (images_root, labels_root):
        for split in ("train", "val"):
            split_path = root / split
            if overwrite and split_path.exists():
                shutil.rmtree(split_path)
            split_path.mkdir(parents=True, exist_ok=True)


def _extract_point(pt: Any) -> tuple[float, float]:
    if isinstance(pt, dict):
        for key in ("x", "X"):
            if key in pt:
                x = float(pt[key])
                break
        else:
            raise ValueError("Point dictionary missing 'x' coordinate")
        for key in ("y", "Y"):
            if key in pt:
                y = float(pt[key])
                break
        else:
            raise ValueError("Point dictionary missing 'y' coordinate")
        return x, y
    if isinstance(pt, Iterable) and not isinstance(pt, (str, bytes)):
        seq = list(pt)
        if len(seq) < 2:
            raise ValueError("Point iterable must contain at least two values")
        return float(seq[0]), float(seq[1])
    raise TypeError(f"Unsupported point type: {type(pt)!r}")


def _compute_box(points: Iterable[Any]) -> tuple[float, float, float, float] | None:
    pts = list(points)
    if len(pts) < 2:
        return None
    xs, ys = [], []
    for pt in pts:
        try:
            x, y = _extract_point(pt)
        except (TypeError, ValueError) as exc:
            logging.debug("Skipping malformed point %s: %s", pt, exc)
            continue
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return None
    x_center = min_x + width / 2.0
    y_center = min_y + height / 2.0
    return x_center, y_center, width, height


def convert_annotations(json_path: Path, image_size: int) -> list[str]:
    if not json_path.exists():
        logging.debug("Annotation file missing for %s", json_path)
        return []
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse JSON %s: %s", json_path, exc)
        return []

    spots = payload.get("spots") or payload.get("spot") or payload.get("annotations") or []
    lines: list[str] = []
    for spot in spots:
        label = str(spot.get("label", "")).strip()
        if label not in LABEL_MAP:
            logging.debug("Skipping unknown label '%s' in %s", label, json_path)
            continue
        box = _compute_box(spot.get("points", []))
        if not box:
            logging.debug("Skipping spot with invalid points in %s", json_path)
            continue
        x_c, y_c, width, height = box
        x_c /= image_size
        y_c /= image_size
        width /= image_size
        height /= image_size
        x_c = max(0.0, min(1.0, x_c))
        y_c = max(0.0, min(1.0, y_c))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        cls = LABEL_MAP[label]
        lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f}")
    return lines


def write_label_file(label_lines: list[str], label_path: Path) -> None:
    if not label_lines:
        if label_path.exists():
            label_path.unlink()
        return
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def prepare_dataset(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    samples = find_samples(args.source)
    if not samples:
        logging.info("No samples to process.")
        return

    ensure_dirs(args.output, overwrite=args.overwrite)

    stats = {"train": 0, "val": 0, "labels": 0}
    for idx, bmp_path in enumerate(samples):
        subset = "val" if idx % GROUP_SIZE >= GROUP_SIZE - VAL_RATIO else "train"
        dest_img = args.output / "images" / subset / bmp_path.name
        copy_image(bmp_path, dest_img)
        stats[subset] += 1

        json_path = bmp_path.with_suffix(".json")
        label_lines = convert_annotations(json_path, args.image_size)
        if label_lines:
            label_path = args.output / "labels" / subset / (bmp_path.stem + ".txt")
            write_label_file(label_lines, label_path)
            stats["labels"] += 1

    logging.info(
        "Dataset preparation complete. Train images: %d, Val images: %d, Label files: %d",
        stats["train"],
        stats["val"],
        stats["labels"],
    )


def main() -> None:
    args = parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
