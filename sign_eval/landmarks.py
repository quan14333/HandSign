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
HAND_REGIONS = ("left_hand", "right_hand")
DEFAULT_REQUIRED_REGIONS = ("left_hand", "right_hand", "face")


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


def tracking_report(
    sequence: LandmarkSequence,
    required_regions: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
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
    label_aware = required_regions is not None
    selected_regions = tuple(required_regions or DEFAULT_REQUIRED_REGIONS)
    required_hands = [region for region in selected_regions if region in HAND_REGIONS]
    hand_columns = {"left_hand": 0, "right_hand": 1}
    if label_aware and required_hands:
        required_hand_rate = float(
            validity[:, [hand_columns[region] for region in required_hands]]
            .all(axis=1)
            .mean()
        )
    else:
        required_hand_rate = hand_rate
    # Coverage is diagnostic only: low percentages do not block evaluation.
    quality_threshold = 0.50 if label_aware else 0.70
    status = "ok" if required_hand_rate >= quality_threshold else "low_quality"
    return {
        "status": status,
        "frame_count": int(len(landmarks)),
        "hand_detected_fraction": round(hand_rate, 4),
        "left_hand_detected_fraction": round(left_rate, 4),
        "right_hand_detected_fraction": round(right_rate, 4),
        "face_detected_fraction": round(face_rate, 4),
        "required_hand_detected_fraction": round(required_hand_rate, 4),
        "required_regions": list(selected_regions),
        "warning": required_hand_rate < 0.70,
    }


def flatten(sequence: np.ndarray) -> np.ndarray:
    return validate_landmarks(sequence).reshape(len(sequence), -1).astype(np.float32)


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def region_indices(regions: tuple[str, ...] | list[str]) -> np.ndarray:
    """Return landmark indexes for a validated ordered collection of regions."""

    unknown = [region for region in regions if region not in REGIONS]
    if unknown:
        raise ValueError(f"Unknown landmark regions: {unknown}")
    if not regions:
        raise ValueError("At least one landmark region is required.")
    return np.concatenate([REGIONS[region] for region in regions])


def calculate_dtw(
    user: np.ndarray,
    reference: np.ndarray,
    regions: tuple[str, ...] | list[str] | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Return path-length-normalized DTW distance and alignment path."""

    user = validate_landmarks(user, "user")
    reference = validate_landmarks(reference, "reference")
    if regions is not None:
        indices = region_indices(regions)
        user_vectors = user[:, indices].reshape(len(user), -1)
        reference_vectors = reference[:, indices].reshape(len(reference), -1)
    else:
        user_vectors = flatten(user)
        reference_vectors = flatten(reference)
    distance, path = fastdtw(user_vectors, reference_vectors, dist=frame_distance)
    if not path:
        raise ValueError("DTW returned an empty alignment path.")
    return float(distance / len(path)), path


def region_distances_on_path(
    user: np.ndarray,
    reference: np.ndarray,
    path: list[tuple[int, int]],
    regions: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return each region's mean and its per-user-frame aligned errors."""

    means: dict[str, float] = {}
    traces: dict[str, np.ndarray] = {}
    selected_regions = tuple(regions or REGIONS)
    region_indices(selected_regions)
    for name in selected_regions:
        indices = REGIONS[name]
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


def region_presence_fraction(sequence: np.ndarray, region: str) -> float:
    """Estimate detector presence in legacy normalized reference arrays.

    Missing regions were stored as 21 identical zero points before global
    normalization.  They therefore still have no within-frame spatial spread,
    unlike a detected hand even when that hand is held still.
    """

    sequence = validate_landmarks(sequence)
    indices = REGIONS[region]
    points = sequence[:, indices]
    spread = np.linalg.norm(points - points.mean(axis=1, keepdims=True), axis=2).mean(axis=1)
    return float(np.mean(spread > 1e-5))


def infer_required_regions(
    references: list[np.ndarray],
    minimum_presence: float = 0.10,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Infer which hands a label uses from its legacy reference sequences.

    A median is used so sporadic false detections do not turn a one-hand sign
    into a two-hand sign.  Face remains part of the score because the corpus
    uses it as the spatial anchor for hand position.
    """

    if not references:
        raise ValueError("At least one reference is required to infer active regions.")
    presence = {
        region: float(
            np.median([region_presence_fraction(reference, region) for reference in references])
        )
        for region in HAND_REGIONS
    }
    active_hands = [
        region for region in HAND_REGIONS if presence[region] >= minimum_presence
    ]
    if not active_hands:
        active_hands = [max(HAND_REGIONS, key=presence.get)]
    return tuple([*active_hands, "face"]), presence


def largest_error_segment(trace: np.ndarray, threshold: float) -> tuple[int, int] | None:
    """Find the longest contiguous user-frame range over the error threshold."""

    over_threshold = np.nan_to_num(trace, nan=-np.inf) > threshold
    if not over_threshold.any():
        return None
    starts = np.flatnonzero(over_threshold & ~np.r_[False, over_threshold[:-1]])
    ends = np.flatnonzero(over_threshold & ~np.r_[over_threshold[1:], False])
    start, end = max(zip(starts, ends), key=lambda item: item[1] - item[0])
    return int(start), int(end)
