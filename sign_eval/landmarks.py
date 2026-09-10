"""Landmark validation, normalization, and DTW primitives.

The reference set already contains z-score-normalized landmarks.  New captures
therefore retain that representation for backwards compatibility while also
persisting tracking masks so missing detections are no longer invisible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastdtw import fastdtw


LANDMARK_SHAPE = (48, 2)
LEFT_HAND_IDX = np.arange(0, 21)
RIGHT_HAND_IDX = np.arange(21, 42)
FACE_IDX = np.arange(42, 48)
REGIONS = {
    "left_hand": LEFT_HAND_IDX,
    "right_hand": RIGHT_HAND_IDX,
    "face": FACE_IDX,
}


@dataclass(frozen=True)
class LandmarkSequence:
    """A sequence plus optional detector-validity information."""

    landmarks: np.ndarray
    validity: np.ndarray | None = None
    source: str | None = None


def validate_landmarks(sequence: np.ndarray, name: str = "sequence") -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != LANDMARK_SHAPE:
        raise ValueError(
            f"{name} must have shape (frames, 48, 2); received {array.shape}."
        )
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two frames.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def load_landmark_sequence(path: str | Path) -> LandmarkSequence:
    """Load legacy ``.npy`` data or a new capture ``.npz`` with masks."""

    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "landmarks" not in loaded:
                raise ValueError(f"{path} has no 'landmarks' array.")
            landmarks = validate_landmarks(loaded["landmarks"], str(path))
            validity = loaded["validity"] if "validity" in loaded else None
        finally:
            loaded.close()
        if validity is not None:
            validity = np.asarray(validity, dtype=bool)
            if validity.shape != (len(landmarks), 3):
                raise ValueError(
                    f"{path} validity must have shape ({len(landmarks)}, 3); "
                    f"received {validity.shape}."
                )
        return LandmarkSequence(landmarks=landmarks, validity=validity, source=str(path))

    return LandmarkSequence(
        landmarks=validate_landmarks(loaded, str(path)), source=str(path)
    )


def zscore_normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Match the normalization used by the existing reference landmarks."""

    sequence = validate_landmarks(sequence)
    means = sequence.mean(axis=(0, 1), keepdims=True)
    scales = sequence.std(axis=(0, 1), keepdims=True)
    scales = np.maximum(scales, 1e-8)
    return ((sequence - means) / scales).astype(np.float32)


def trim_to_active_hands(
    landmarks: np.ndarray, validity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trim leading/trailing frames without either detected hand."""

    if validity.shape != (len(landmarks), 3):
        raise ValueError("validity must have one left/right/face value per frame.")
    active = validity[:, :2].any(axis=1)
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        raise ValueError("No hand was detected in the recording.")
    start, end = int(indices[0]), int(indices[-1]) + 1
    return landmarks[start:end], validity[start:end]


def tracking_report(sequence: LandmarkSequence) -> dict[str, Any]:
    """Report capture quality without pretending legacy data has masks."""

    landmarks = sequence.landmarks
    zero_frames = np.all(landmarks == 0, axis=(1, 2))
    if np.all(zero_frames):
        return {
            "status": "invalid",
            "reason": "All landmark frames are zero.",
            "frame_count": int(len(landmarks)),
        }

    if sequence.validity is None:
        return {
            "status": "unknown",
            "reason": "Legacy .npy data has no detector-validity mask.",
            "frame_count": int(len(landmarks)),
            "zero_frame_fraction": round(float(zero_frames.mean()), 4),
        }

    validity = sequence.validity
    left_rate = float(validity[:, 0].mean())
    right_rate = float(validity[:, 1].mean())
    face_rate = float(validity[:, 2].mean())
    hand_rate = float(validity[:, :2].any(axis=1).mean())
    status = "ok" if hand_rate >= 0.70 else "low_quality"
    return {
        "status": status,
        "frame_count": int(len(landmarks)),
        "hand_detected_fraction": round(hand_rate, 4),
        "left_hand_detected_fraction": round(left_rate, 4),
        "right_hand_detected_fraction": round(right_rate, 4),
        "face_detected_fraction": round(face_rate, 4),
    }


def flatten(sequence: np.ndarray) -> np.ndarray:
    return validate_landmarks(sequence).reshape(len(sequence), -1).astype(np.float32)


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def calculate_dtw(
    user: np.ndarray, reference: np.ndarray
) -> tuple[float, list[tuple[int, int]]]:
    """Return path-length-normalized DTW distance and alignment path."""

    distance, path = fastdtw(flatten(user), flatten(reference), dist=frame_distance)
    if not path:
        raise ValueError("DTW returned an empty alignment path.")
    return float(distance / len(path)), path


def region_distances_on_path(
    user: np.ndarray,
    reference: np.ndarray,
    path: list[tuple[int, int]],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return each region's mean and its per-user-frame aligned errors."""

    means: dict[str, float] = {}
    traces: dict[str, np.ndarray] = {}
    for name, indices in REGIONS.items():
        values_by_frame: list[list[float]] = [[] for _ in range(len(user))]
        for user_index, ref_index in path:
            distance = np.linalg.norm(
                user[user_index, indices] - reference[ref_index, indices], axis=1
            ).mean()
            values_by_frame[user_index].append(float(distance))
        trace = np.array(
            [np.mean(values) if values else np.nan for values in values_by_frame],
            dtype=np.float32,
        )
        means[name] = float(np.nanmean(trace))
        traces[name] = trace
    return means, traces


def largest_error_segment(trace: np.ndarray, threshold: float) -> tuple[int, int] | None:
    """Find the longest contiguous user-frame range over the error threshold."""

    over_threshold = np.nan_to_num(trace, nan=-np.inf) > threshold
    if not over_threshold.any():
        return None
    starts = np.flatnonzero(over_threshold & ~np.r_[False, over_threshold[:-1]])
    ends = np.flatnonzero(over_threshold & ~np.r_[over_threshold[1:], False])
    start, end = max(zip(starts, ends), key=lambda item: item[1] - item[0])
    return int(start), int(end)
