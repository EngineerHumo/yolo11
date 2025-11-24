"""Export a trained YOLO model to ONNX and run ONNX inference mirroring predict_to_txt.py."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

# Paths for training weights, export target, and ONNX inference source/output
TRAINED_MODEL_PATH = Path("/home/wensheng/jiaqi/ultralytics/runs/detect/train65/weights/best.pt")
ONNX_MODEL_PATH = Path("/home/wensheng/jiaqi/ultralytics/runs/detect/train65/weights/best.onnx")
ONNX_SOURCE_DIR = Path("/home/wensheng/jiaqi/ultralytics/data/train/image")
ONNX_OUTPUT_DIR = Path("/home/wensheng/jiaqi/ultralytics/output_onnx")

# Preprocess to align with predict_to_txt.py
TARGET_IMAGE_SIZE = 1240


def prepare_image(image: np.ndarray) -> np.ndarray:
    """Resize images to the square size expected by the training pipeline."""
    if image.shape[0] != TARGET_IMAGE_SIZE or image.shape[1] != TARGET_IMAGE_SIZE:
        return cv2.resize(
            image,
            (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
    return image


def collect_images(source_dir: Path) -> list[Path]:
    images: list[Path] = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not images:
        raise FileNotFoundError(f"No JPG images found in {source_dir!s}")
    return images


def draw_detections(image: np.ndarray, detections) -> np.ndarray:
    """Draw bounding boxes with class labels and confidence on the image."""
    colors = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    )

    annotated = image.copy()
    for result in detections:
        boxes = getattr(result.boxes, "xyxy", None)
        class_ids = getattr(result.boxes, "cls", None)
        confidences = getattr(result.boxes, "conf", None)
        if boxes is None or class_ids is None or confidences is None:
            continue

        boxes_np = boxes.cpu().numpy()
        class_ids_list = class_ids.int().cpu().tolist()
        confidences_list = confidences.cpu().tolist()

        for class_id, conf, (x1, y1, x2, y2) in zip(class_ids_list, confidences_list, boxes_np):
            color = colors[class_id % len(colors)]
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{class_id}:{conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (int(x1), max(0, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return annotated


def save_detections_to_txt(output_path: Path, detections) -> None:
    """Persist detections in YOLO txt format including confidence for each box."""
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        for result in detections:
            boxes = getattr(result.boxes, "xywhn", None)
            class_ids = getattr(result.boxes, "cls", None)
            confidences = getattr(result.boxes, "conf", None)
            if boxes is None or class_ids is None or confidences is None:
                continue

            boxes_np = boxes.cpu().numpy()
            class_ids_list = class_ids.int().cpu().tolist()
            confidences_list = confidences.cpu().tolist()

            for class_id, conf, (x_center, y_center, width, height) in zip(class_ids_list, confidences_list, boxes_np):
                file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")


def save_video_from_images(image_paths: list[Path], output_dir: Path) -> None:
    if not image_paths:
        return

    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise RuntimeError(f"Failed to read image {image_paths[0]!s}")

    height, width = first_image.shape[:2]
    video_path = output_dir / "output.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            writer.write(frame)
    finally:
        writer.release()


def export_to_onnx() -> Path:
    """Export the trained YOLO model to ONNX with post-processing embedded."""
    model = YOLO(TRAINED_MODEL_PATH)
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=TARGET_IMAGE_SIZE,
            opset=13,
            nms=True,
        )
    )

    if exported_path != ONNX_MODEL_PATH:
        ONNX_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(ONNX_MODEL_PATH)
        exported_path = ONNX_MODEL_PATH

    return exported_path


def run_onnx_inference(onnx_path: Path, images: list[Path]) -> None:
    """Run ONNX inference using the Ultralytics ONNX backend and save detections."""
    onnx_model = YOLO(onnx_path)
    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_images: list[Path] = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image {image_path!s}")
        processed_image = prepare_image(image)
        detections = onnx_model.predict(source=processed_image, verbose=False, save=False)
        annotated_image = draw_detections(processed_image, detections)
        output_image_path = ONNX_OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_image_path), annotated_image)
        saved_images.append(output_image_path)
        save_detections_to_txt(output_image_path, detections)

    save_video_from_images(saved_images, ONNX_OUTPUT_DIR)


def main() -> None:
    onnx_path = export_to_onnx()
    images = collect_images(ONNX_SOURCE_DIR)
    run_onnx_inference(onnx_path, images)
    print(
        "Exported model to ONNX, ran inference, and saved annotated images, txt files, and video to",
        f"{ONNX_OUTPUT_DIR!s}.",
    )


if __name__ == "__main__":
    main()
