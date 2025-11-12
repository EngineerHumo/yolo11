"""Predict script for running YOLO detections on a directory of images.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "/home/wensheng/jiaqi/ultralytics/runs/detect/train25/weights/best.pt"
DEFAULT_SOURCE_DIR = "/home/wensheng/jiaqi/ultralytics/video"


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


def compute_reference_hsv_mean(image_paths: Sequence[Path], max_frames: int = 100) -> Tuple[float, float, float]:
    if not image_paths:
        raise ValueError("No images provided for computing HSV mean.")

    hsv_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for image_path in image_paths[:max_frames]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image {image_path!s}")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_sum += hsv.reshape(-1, 3).sum(axis=0)
        total_pixels += hsv.shape[0] * hsv.shape[1]

    if total_pixels == 0:
        raise RuntimeError("Images have zero pixels; cannot compute HSV mean.")

    hsv_mean = hsv_sum / float(total_pixels)
    return float(hsv_mean[0]), float(hsv_mean[1]), float(hsv_mean[2])


def adjust_v_channel_to_mean(image: np.ndarray, target_v_mean: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    current_v = hsv[:, :, 2]
    current_mean = float(current_v.mean())

    if current_mean <= 0.0:
        adjusted_v = np.clip(np.full_like(current_v, target_v_mean), 0.0, 255.0)
    else:
        scale = target_v_mean / current_mean
        adjusted_v = np.clip(current_v * scale, 0.0, 255.0)

    hsv[:, :, 2] = adjusted_v
    adjusted_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted_image


def run_detection_and_save(
    model: YOLO,
    image: np.ndarray,
    output_image_path: Path,
    output_txt_path: Path,
) -> np.ndarray:
    cropped = crop_bottom_rows(image)
    detections = model.predict(source=cropped, verbose=False, save=False)

    resized, scale_x, scale_y = ensure_size(cropped, 1240, 1240)
    annotated = resized

    names: Sequence[str] | dict[int, str]
    names = detections[0].names if detections else {}

    image_classes: Set[int] = set()
    yolo_records: List[str] = []
    height, width = cropped.shape[:2]

    for result in detections:
        if result.boxes is None or result.boxes.xyxy is None or result.boxes.cls is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.int().cpu().tolist()

        annotated, present = draw_boxes(annotated, boxes, class_ids, scale_x, scale_y)
        image_classes.update(present)

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = box.astype(float)
            box_width = x2 - x1
            box_height = y2 - y1
            x_center = x1 + box_width / 2.0
            y_center = y1 + box_height / 2.0

            if width > 0 and height > 0:
                x_center /= width
                y_center /= height
                box_width /= width
                box_height /= height

            yolo_records.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

    annotated = draw_legend(annotated, image_classes, names)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image_path), annotated):
        raise RuntimeError(f"Failed to write annotated image to {output_image_path!s}")

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(yolo_records))

    return annotated


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    images = collect_images(source_dir)

    reference_count = min(len(images), 100)
    reference_hsv_mean = compute_reference_hsv_mean(images, max_frames=reference_count)
    print(
        "Reference HSV mean from first "
        f"{reference_count} frame(s): "
        f"H={reference_hsv_mean[0]:.2f}, S={reference_hsv_mean[1]:.2f}, V={reference_hsv_mean[2]:.2f}"
    )

    model = YOLO(args.model)

    adjust_dir = Path("/video/output/adjust")
    originmask_dir = Path("/video/output/originmask")
    adjustmask_dir = Path("/video/output/adjustmask")

    for directory in (adjust_dir, originmask_dir, adjustmask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    target_v_mean = reference_hsv_mean[2]

    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    origin_video_path = originmask_dir / "predict.mp4"
    adjust_video_path = adjustmask_dir / "predict_adjust.mp4"
    origin_writer: cv2.VideoWriter | None = None
    adjust_writer: cv2.VideoWriter | None = None

    try:
        for index, image_path in enumerate(images):
            original = cv2.imread(str(image_path))
            if original is None:
                raise RuntimeError(f"Failed to read image {image_path!s}")

            if index < reference_count:
                adjusted = original.copy()
            else:
                adjusted = adjust_v_channel_to_mean(original, target_v_mean)

            adjusted_filename = f"{image_path.stem}_adjust{image_path.suffix}"
            adjusted_image_path = adjust_dir / adjusted_filename
            if not cv2.imwrite(str(adjusted_image_path), adjusted):
                raise RuntimeError(f"Failed to write adjusted image to {adjusted_image_path!s}")

            origin_output_image = originmask_dir / image_path.name
            origin_output_txt = originmask_dir / f"{image_path.stem}.txt"
            origin_annotated = run_detection_and_save(
                model, original, origin_output_image, origin_output_txt
            )

            if origin_writer is None:
                frame_size = (origin_annotated.shape[1], origin_annotated.shape[0])
                origin_writer = cv2.VideoWriter(
                    str(origin_video_path), fourcc, fps, frame_size
                )
                if not origin_writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {origin_video_path!s}")
            else:
                writer_height = int(round(origin_writer.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                writer_width = int(round(origin_writer.get(cv2.CAP_PROP_FRAME_WIDTH)))
                if (
                    origin_annotated.shape[0] != writer_height
                    or origin_annotated.shape[1] != writer_width
                ):
                    raise RuntimeError("Annotated frame size changed during processing for original video.")

            origin_writer.write(origin_annotated)

            adjust_output_image = adjustmask_dir / adjusted_filename
            adjust_output_txt = adjustmask_dir / f"{image_path.stem}_adjust.txt"
            adjust_annotated = run_detection_and_save(
                model, adjusted, adjust_output_image, adjust_output_txt
            )

            if adjust_writer is None:
                frame_size = (adjust_annotated.shape[1], adjust_annotated.shape[0])
                adjust_writer = cv2.VideoWriter(
                    str(adjust_video_path), fourcc, fps, frame_size
                )
                if not adjust_writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {adjust_video_path!s}")
            else:
                writer_height = int(round(adjust_writer.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                writer_width = int(round(adjust_writer.get(cv2.CAP_PROP_FRAME_WIDTH)))
                if (
                    adjust_annotated.shape[0] != writer_height
                    or adjust_annotated.shape[1] != writer_width
                ):
                    raise RuntimeError("Annotated frame size changed during processing for adjusted video.")

            adjust_writer.write(adjust_annotated)
    finally:
        if origin_writer is not None:
            origin_writer.release()
        if adjust_writer is not None:
            adjust_writer.release()

    print("Processing complete.")
    print(f"Adjusted images saved to: {adjust_dir!s}")
    print(f"Original detections saved to: {originmask_dir!s}")
    print(f"Adjusted detections saved to: {adjustmask_dir!s}")
    print(f"Original detection video: {origin_video_path!s}")
    print(f"Adjusted detection video: {adjust_video_path!s}")


if __name__ == "__main__":
    main()
