"""VideoMAE-based sign-label recognition used before form evaluation."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAME = "star092304/vi-sign-language-videomae-base"
NUM_FRAMES = 16


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    """Load the model lazily so landmark-only evaluation has no torch cost."""

    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME, ignore_mismatched_sizes=True
    )
    feature_count = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.LayerNorm(feature_count), nn.Dropout(0.3), nn.Linear(feature_count, model.config.num_labels)
    )
    checkpoint_path = hf_hub_download(
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
    model.eval()
    return processor, model, torch


def _load_video(video_path: str | Path, frame_count: int) -> list[np.ndarray]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0))
    if len(reader) == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.linspace(0, len(reader) - 1, frame_count).astype(int)
    return list(reader.get_batch(indices).asnumpy())


def predict_video(video_path: str | Path, top_k: int = 5) -> dict[str, Any]:
    """Return calibrated information needed by ``SignEvaluator`` semantics."""

    processor, model, torch = _load_model()
    frames = _load_video(video_path, NUM_FRAMES)
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    count = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=count)
    predictions = [
        {
            "label": model.config.id2label[index.item()],
            "confidence": round(float(value.item()), 6),
        }
        for value, index in zip(values, indices)
    ]
    return {
        "predicted_label": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "margin": round(
            predictions[0]["confidence"] - predictions[1]["confidence"], 6
        ) if len(predictions) > 1 else 1.0,
        "top_predictions": predictions,
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
