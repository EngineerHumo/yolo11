"""Predict script for running YOLO detections on a directory of images.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "/home/wensheng/jiaqi/ultralytics/runs/detect/train25/weights/best.pt"
DEFAULT_SOURCE_DIR = "/home/wensheng/jiaqi/ultralytics/video"
DEFAULT_OUTPUT_SUBDIR = "predictions"
DEFAULT_VIDEO_NAME = "predictions.mp4"
DEFAULT_VIDEO_FPS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detections on a directory of images.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained YOLO model weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing JPG images for detection.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Name of the subdirectory where annotated images will be saved.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default=DEFAULT_VIDEO_NAME,
        help="Name of the output MP4 video file created from annotated images.",
    )
    return parser.parse_args()


def collect_images(source_dir: Path) -> List[Path]:
    images: List[Path] = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not images:
        raise FileNotFoundError(f"No JPG images found in {source_dir!s}")
    return images


def ensure_output_dir(source_dir: Path, output_name: str) -> Path:
    output_dir = source_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


COLOR_PALETTE: Sequence[tuple[int, int, int]] = (
    (255, 255, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 255),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


def crop_bottom_rows(image: np.ndarray, rows_to_remove: int = 38) -> np.ndarray:
    if image.shape[0] <= rows_to_remove:
        raise ValueError(
            f"Image height {image.shape[0]} is not greater than rows_to_remove={rows_to_remove}."
        )
    return image[: image.shape[0] - rows_to_remove, :, :]


def ensure_size(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, float]:
    current_height, current_width = image.shape[:2]
    if current_width == width and current_height == height:
        return image.copy(), 1.0, 1.0

    scale_x = width / current_width
    scale_y = height / current_height
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized, scale_x, scale_y


def draw_legend(
    image: np.ndarray, class_ids: Iterable[int], class_names: Sequence[str] | dict[int, str]
) -> np.ndarray:
    unique_ids: List[int] = sorted(set(class_ids))
    if not unique_ids:
        return image

    legend_padding = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_base = max(image.shape[0], image.shape[1]) / 1000.0
    font_scale = max(0.4, 0.5 * scale_base)
    thickness = max(1, int(round(font_scale * 2)))
    box_edge = int(round(18 * scale_base))
    box_edge = max(box_edge, 12)

    text_widths: List[int] = []
    legend_labels: List[str] = []
    for class_id in unique_ids:
        name = class_names[class_id] if isinstance(class_names, dict) else class_names[class_id]
        legend_labels.append(name)
        (text_width, text_height), _ = cv2.getTextSize(name, font, font_scale, thickness)
        text_widths.append(text_width)

    legend_width = (
        3 * legend_padding
        + box_edge
        + (max(text_widths) if text_widths else 0)
    )
    legend_height = legend_padding + len(unique_ids) * (box_edge + legend_padding)

    start_x = max(image.shape[1] - legend_width - legend_padding, 0)
    start_y = max(image.shape[0] - legend_height - legend_padding, 0)
    end_x = min(start_x + legend_width, image.shape[1])
    end_y = min(start_y + legend_height, image.shape[0])

    overlay = image.copy()
    cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, dst=image)

    current_y = start_y + legend_padding
    for class_id, label in zip(unique_ids, legend_labels):
        color = color_for_class(class_id)
        box_start = (start_x + legend_padding, current_y)
        box_end = (box_start[0] + box_edge, box_start[1] + box_edge)
        cv2.rectangle(image, box_start, box_end, color, -1)

        text_x = box_end[0] + legend_padding
        text_y = box_start[1] + box_edge - max(2, int(0.25 * box_edge))
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        current_y += box_edge + legend_padding

    return image


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: Sequence[int],
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, Set[int]]:
    annotated = image.copy()
    present_classes: Set[int] = set()
    for box, class_id in zip(boxes, classes):
        x1, y1, x2, y2 = box.astype(float)
        x1 = int(round(x1 * scale_x))
        y1 = int(round(y1 * scale_y))
        x2 = int(round(x2 * scale_x))
        y2 = int(round(y2 * scale_y))
        color = color_for_class(int(class_id))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        present_classes.add(int(class_id))
    return annotated, present_classes


def save_annotated_images(model: YOLO, images: List[Path], output_dir: Path) -> List[Path]:
    saved_images: List[Path] = []
    for image_path in images:
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Failed to read image {image_path!s}")

        cropped = crop_bottom_rows(original)
        detections = model.predict(source=cropped, verbose=False, save=False)

        resized, scale_x, scale_y = ensure_size(cropped, 1240, 1240)

        names: Sequence[str] | dict[int, str]
        names = detections[0].names if detections else {}

        annotated = resized
        image_classes: Set[int] = set()
        for result in detections:
            if result.boxes is None or result.boxes.xyxy is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().tolist()
            annotated, present = draw_boxes(annotated, boxes, class_ids, scale_x, scale_y)
            image_classes.update(present)

        annotated = draw_legend(annotated, image_classes, names)

        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to write annotated image to {output_path!s}")
        saved_images.append(output_path)
    return saved_images


def build_video_from_images(images: List[Path], output_path: Path, fps: float) -> None:
    first_image = cv2.imread(str(images[0]))
    if first_image is None:
        raise RuntimeError(f"Unable to read the first annotated image {images[0]!s}")

    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Unable to read annotated image {image_path!s}")
            video_writer.write(frame)
    finally:
        video_writer.release()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = ensure_output_dir(source_dir, args.output_name)

    model = YOLO(args.model)

    images = collect_images(source_dir)
    saved_images = save_annotated_images(model, images, output_dir)

    video_path = output_dir / args.video_name
    build_video_from_images(saved_images, video_path, DEFAULT_VIDEO_FPS)

    print(f"Annotated images saved to: {output_dir!s}")
    print(f"Video saved to: {video_path!s}")


if __name__ == "__main__":
    main()
