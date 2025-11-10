"""Predict script for running YOLO detections on a directory of images.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms as T
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "/home/wensheng/jiaqi/ultralytics/runs/detect/train53/weights/best.pt"
DEFAULT_SOURCE_DIR = "/home/wensheng/jiaqi/ultralytics/video"
DEFAULT_OUTPUT_SUBDIR = "/home/wensheng/jiaqi/ultralytics/output_classification"
DEFAULT_VIDEO_NAME = "predictions.mp4"
DEFAULT_VIDEO_FPS = 5.0


CLASSIFICATION_TO_YOLO_LABEL: dict[int, str] = {
    0: "old",
    1: "1",
    2: "2",
    3: "2+",
    4: "3",
    5: "3+",
}


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: Tensor, label: Tensor) -> Tensor:  # pragma: no cover - training API
        raise RuntimeError("ArcMarginProduct forward is not supported for inference usage.")

    def inference(self, embeddings: Tensor) -> Tensor:
        normalized_embeddings = F.normalize(embeddings)
        normalized_weights = F.normalize(self.weight)
        return torch.matmul(normalized_embeddings, normalized_weights.t()) * self.s


class _ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output_dim = base_channels * 4

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class DualEncoderMetricModel(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.spot_encoder = _ConvEncoder(in_channels=3)
        self.global_encoder = _ConvEncoder(in_channels=3)
        combined_dim = self.spot_encoder.output_dim + self.global_encoder.output_dim
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, spot: Tensor, global_image: Tensor) -> Tuple[Tensor, Tensor]:
        spot_feat = self.spot_encoder(spot)
        global_feat = self.global_encoder(global_image)
        combined = torch.cat([spot_feat, global_feat], dim=1)
        embedding = self.projection(combined)
        embedding = F.normalize(embedding, dim=1)
        return embedding, combined


def _build_transforms() -> Tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spot_transform = T.Compose([T.ToTensor(), normalize])
    global_transform = T.Compose([T.ToTensor(), normalize])
    return spot_transform, global_transform


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[DualEncoderMetricModel, ArcMarginProduct]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DualEncoderMetricModel().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)  # type: ignore[index]

    arcface_state = checkpoint["arcface_state"]  # type: ignore[index]
    out_features, in_features = arcface_state["weight"].shape
    arcface = ArcMarginProduct(in_features=in_features, out_features=out_features).to(device)
    arcface.load_state_dict(arcface_state, strict=False)

    model.eval()
    arcface.eval()
    return model, arcface


@dataclass
class ClassificationComponents:
    model: DualEncoderMetricModel
    arcface: ArcMarginProduct
    spot_transform: T.Compose
    global_transform: T.Compose
    device: torch.device


def _convert_bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def _extract_center_patch(image: np.ndarray, box: np.ndarray, size: int = 64) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = size / 2.0

    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + size
    bottom = top + size

    h, w = image.shape[:2]
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    patch = image[top:bottom, left:right]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return patch


def _average_pool_to_size(image: np.ndarray, size: int = 128) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(tensor, (size, size))
    pooled = pooled.squeeze(0).permute(1, 2, 0)
    pooled = pooled.clamp(0, 255).byte().cpu().numpy()
    return pooled


def classify_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    components: ClassificationComponents,
) -> List[int]:
    if boxes.size == 0:
        return []

    spot_tensors: List[Tensor] = []
    for box in boxes:
        patch = _extract_center_patch(image, box, size=64)
        pil_image = _convert_bgr_to_pil(patch)
        spot_tensor = components.spot_transform(pil_image)
        spot_tensors.append(spot_tensor)

    global_image = _average_pool_to_size(image, size=128)
    global_tensor = components.global_transform(_convert_bgr_to_pil(global_image))

    spot_batch = torch.stack(spot_tensors, dim=0).to(components.device)
    global_batch = global_tensor.unsqueeze(0).to(components.device)
    global_batch = global_batch.expand(spot_batch.shape[0], -1, -1, -1).contiguous()

    with torch.no_grad():
        outputs = components.model(spot_batch, global_batch)
        if isinstance(outputs, tuple):
            embedding = outputs[0]
        else:
            embedding = outputs
        scores = components.arcface.inference(embedding).detach().cpu()

    predicted = scores.argmax(dim=1).tolist()
    return predicted


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
    parser.add_argument(
        "--classification-checkpoint",
        type=str,
        default="/home/wensheng/jiaqi/ultralytics/classification.pt",
        help="Path to the classification.pt checkpoint for the spot classifier.",
    )
    parser.add_argument(
        "--classification-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the classification model on (e.g. 'cpu' or 'cuda').",
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
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
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


def class_label(class_names: Sequence[str] | dict[int, str], class_id: int) -> str:
    if isinstance(class_names, dict):
        return class_names.get(class_id, f"class {class_id}")
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    print(class_id)
    return f"class {class_id}"


def _build_name_to_id(names: Sequence[str] | dict[int, str]) -> dict[str, int]:
    if isinstance(names, dict):
        return {value: key for key, value in names.items()}
    return {name: idx for idx, name in enumerate(names)}


def map_classification_to_yolo(
    predicted: Sequence[int],
    names: Sequence[str] | dict[int, str],
) -> tuple[List[int], List[str]]:
    name_to_id = _build_name_to_id(names)
    mapped_classes: List[int] = []
    labels: List[str] = []
    for cls_id in predicted:
        label = CLASSIFICATION_TO_YOLO_LABEL.get(cls_id, f"class {cls_id}")
        mapped_class = name_to_id.get(label)
        if mapped_class is None:
            mapped_class = cls_id
        mapped_classes.append(mapped_class)
        labels.append(label)
    return mapped_classes, labels


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
        name = class_label(class_names, class_id)
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
    labels: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, Set[int]]:
    annotated = image.copy()
    present_classes: Set[int] = set()
    for idx, (box, class_id) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = box.astype(float)
        x1 = int(round(x1 * scale_x))
        y1 = int(round(y1 * scale_y))
        x2 = int(round(x2 * scale_x))
        y2 = int(round(y2 * scale_y))
        color = color_for_class(int(class_id))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if labels is not None and idx < len(labels):
            label_text = labels[idx]
            if label_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(image.shape[0], image.shape[1]) / 1000.0
                font_scale = max(font_scale, 0.5)
                thickness = max(1, int(round(font_scale * 2)))
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )
                text_origin = (x1, max(y1 - 5, 0))
                box_start = (text_origin[0], text_origin[1] - text_height - baseline)
                box_end = (text_origin[0] + text_width, text_origin[1])
                box_start = (max(box_start[0], 0), max(box_start[1], 0))
                box_end = (
                    min(box_end[0], annotated.shape[1] - 1),
                    min(box_end[1], annotated.shape[0] - 1),
                )
                cv2.rectangle(annotated, box_start, box_end, color, -1)
                cv2.putText(
                    annotated,
                    label_text,
                    (box_start[0], box_end[1] - baseline),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
        present_classes.add(int(class_id))
    return annotated, present_classes


def save_annotated_images(
    model: YOLO,
    images: List[Path],
    output_dir: Path,
    classification: Optional[ClassificationComponents] = None,
) -> Tuple[List[Path], List[Path]]:
    saved_images: List[Path] = []
    saved_combined: List[Path] = []
    for image_path in images:
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Failed to read image {image_path!s}")

        cropped = crop_bottom_rows(original)
        detections = model.predict(source=cropped, verbose=False, save=False)

        resized, scale_x, scale_y = ensure_size(cropped, 1240, 1240)

        names: Sequence[str] | dict[int, str]
        names = detections[0].names if detections else {}

        annotated = resized.copy()
        image_classes: Set[int] = set()
        detection_boxes: List[np.ndarray] = []
        for result in detections:
            if result.boxes is None or result.boxes.xyxy is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().tolist()
            if boxes.size:
                detection_boxes.append(boxes)
            annotated, present = draw_boxes(annotated, boxes, class_ids, scale_x, scale_y)
            image_classes.update(present)

        annotated = draw_legend(annotated, image_classes, names)

        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to write annotated image to {output_path!s}")
        saved_images.append(output_path)

        if classification is not None:
            combined_image = resized.copy()
            combined_classes: Set[int] = set()
            combined_boxes = (
                np.vstack(detection_boxes)
                if detection_boxes
                else np.empty((0, 4), dtype=np.float32)
            )
            if combined_boxes.size:
                predicted_classes = classify_detections(cropped, combined_boxes, classification)
                mapped_classes, labels = map_classification_to_yolo(predicted_classes, names)
                combined_image, present = draw_boxes(
                    combined_image,
                    combined_boxes,
                    mapped_classes,
                    scale_x,
                    scale_y,
                    labels=labels,
                )
                combined_classes.update(present)
            combined_image = draw_legend(combined_image, combined_classes, names)
            combine_output_path = output_dir / f"{image_path.stem}_combine{image_path.suffix}"
            if not cv2.imwrite(str(combine_output_path), combined_image):
                raise RuntimeError(
                    f"Failed to write combined annotated image to {combine_output_path!s}"
                )
            saved_combined.append(combine_output_path)
    return saved_images, saved_combined


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

    classification_device = torch.device(args.classification_device)
    classification_checkpoint = Path(args.classification_checkpoint)
    spot_transform, global_transform = _build_transforms()
    classification_model, arcface = load_checkpoint(classification_checkpoint, classification_device)
    classification_components = ClassificationComponents(
        model=classification_model,
        arcface=arcface,
        spot_transform=spot_transform,
        global_transform=global_transform,
        device=classification_device,
    )

    images = collect_images(source_dir)
    saved_images, saved_combined_images = save_annotated_images(
        model, images, output_dir, classification=classification_components
    )

    video_path = output_dir / args.video_name
    build_video_from_images(saved_images, video_path, DEFAULT_VIDEO_FPS)

    combined_video_path = output_dir / (
        f"{Path(args.video_name).stem}_combine{Path(args.video_name).suffix}"
    )
    if saved_combined_images:
        build_video_from_images(saved_combined_images, combined_video_path, DEFAULT_VIDEO_FPS)

    print(f"Annotated images saved to: {output_dir!s}")
    print(f"Video saved to: {video_path!s}")
    if saved_combined_images:
        print(f"Combined annotated images saved to: {output_dir!s}")
        print(f"Combined video saved to: {combined_video_path!s}")


if __name__ == "__main__":
    main()
