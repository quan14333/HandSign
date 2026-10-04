

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAME = "star092304/vi-sign-language-videomae-base"
LOCAL_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "recognition_model"
NUM_FRAMES = 16


class ModelSetupError(RuntimeError):
    """An actionable recognition setup error safe to show in the practice UI."""


def model_source() -> tuple[str, bool]:
    configured = os.environ.get("HANDSIGN_MODEL_PATH")
    local_path = Path(configured) if configured else LOCAL_MODEL_PATH
    if configured or local_path.is_dir():
        required = ("config.json", "preprocessor_config.json", "classifier_sequential.pth")
        if not all((local_path / name).is_file() for name in required) or not any(
            (local_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
        ):
            raise ModelSetupError("Model HandSign cục bộ chưa đầy đủ. Hãy chạy python scripts/download_model.py trong thư mục HandSign rồi khởi động lại dịch vụ.")
        return str(local_path), True
    return MODEL_NAME, False

# The published checkpoint has several mojibake label strings in config.json.
# Keep the checkpoint's class IDs unchanged and repair display text only.
LABEL_TEXT_CORRECTIONS = {
    1: "Ban ngày",
    5: "Bàn tay",
    14: "Chúng ta",
    15: "Chân",
    16: "Chào",
    27: "Cá",
    28: "Cách ly",
    31: "Ghét",
    32: "Giúp",
    36: "Hôm nay",
    39: "Khai báo",
    40: "Khu cách ly",
    41: "Khóc",
    50: "Ngón tay",
    51: "Nhà",
    52: "Nhìn",
    55: "Nói",
    56: "Nôn ói",
    59: "Phía sau",
    69: "Thích",
    77: "Tôi",
    84: "Xe máy",
    87: "Xin phép",
}


def _repair_label_text(model: Any) -> None:
    """Repair corrupted Vietnamese text without changing checkpoint class IDs."""

    id2label = {int(index): label for index, label in model.config.id2label.items()}
    for index, label in LABEL_TEXT_CORRECTIONS.items():
        if index in id2label:
            id2label[index] = label
    model.config.id2label = id2label
    model.config.label2id = {label: index for index, label in id2label.items()}


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    """Load the model lazily so landmark-only evaluation has no torch cost."""

    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    source, use_local = model_source()
    try:
        processor = VideoMAEImageProcessor.from_pretrained(source, local_files_only=use_local)
        model = VideoMAEForVideoClassification.from_pretrained(
            source, ignore_mismatched_sizes=True, local_files_only=use_local
        )
    except OSError as error:
        raise ModelSetupError("Không nạp được model nhận diện. Hãy chạy python scripts/download_model.py trong thư mục HandSign khi có mạng, rồi khởi động lại dịch vụ.") from error
    feature_count = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.LayerNorm(feature_count), nn.Dropout(0.3), nn.Linear(feature_count, model.config.num_labels)
    )
    checkpoint_path = Path(source) / "classifier_sequential.pth" if use_local else hf_hub_download(
        repo_id=MODEL_NAME, filename="classifier_sequential.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    converted: dict[str, Any] = {}
    for key, value in checkpoint.items():
        if key.endswith(".q_bias"):
            converted[key.replace(".q_bias", ".query.bias")] = value
        elif key.endswith(".v_bias"):
            converted[key.replace(".v_bias", ".value.bias")] = value
        else:
            converted[key] = value
    for index in range(model.config.num_hidden_layers):
        name = f"videomae.encoder.layer.{index}.attention.attention.key.bias"
        weight_name = f"videomae.encoder.layer.{index}.attention.attention.key.weight"
        converted[name] = torch.zeros(
            model.state_dict()[weight_name].shape[0], dtype=model.state_dict()[weight_name].dtype
        )
    model.load_state_dict(converted, strict=False)
    _repair_label_text(model)
    model.eval()
    return processor, model, torch


def _load_video(video_path: str | Path, frame_count: int) -> list[np.ndarray]:
    """Sample RGB frames with the same OpenCV backend used by capture/playback."""
    import cv2

    reader = cv2.VideoCapture(str(video_path))
    try:
        if not reader.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or frame_count <= 0:
            raise ValueError(f"Video has no readable frame count: {video_path}")
        indices = np.linspace(0, total - 1, frame_count).astype(int)
        frames = []
        sample = 0
        for index in range(total):
            ok, frame = reader.read()
            if not ok:
                raise ValueError(f"Unable to decode frame {index} of {video_path}")
            if index == indices[sample]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                while sample < frame_count and indices[sample] == index:
                    frames.append(rgb.copy())
                    sample += 1
            if sample == frame_count:
                break
        return frames
    finally:
        reader.release()


def predict_video(video_path: str | Path, top_k: int = 5) -> dict[str, Any]:
    """Return model probabilities and the true top-two gap (not calibrated)."""

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    processor, model, torch = _load_model()
    frames = _load_video(video_path, NUM_FRAMES)
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    # Always inspect the runner-up, even if the caller only displays top one.
    count = min(max(top_k, 2), probabilities.numel())
    values, indices = torch.topk(probabilities, k=count)
    predictions = [
        {
            "label": model.config.id2label[index.item()],
            "confidence": float(value.item()),
        }
        for value, index in zip(values, indices)
    ]
    return {
        "predicted_label": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "margin": (
            predictions[0]["confidence"] - predictions[1]["confidence"]
        ) if len(predictions) > 1 else None,
        "top_predictions": predictions[:top_k],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize a Vietnamese sign from video.")
    parser.add_argument("video", nargs="?", default="data/user/video_user/trimmed_video.mp4")
    args = parser.parse_args()
    result = predict_video(args.video)
    for rank, prediction in enumerate(result["top_predictions"], start=1):
        print(f"{rank}. {prediction['label']} -> {prediction['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
