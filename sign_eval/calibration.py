"""Build label-specific calibration data from the reference corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .landmarks import load_landmark_sequence


DEFAULT_REFERENCE_ROOT = Path("data/landmarks (1)")
DEFAULT_PAIR_INFO = Path("result2/info.npy")
DEFAULT_OUTPUT = Path("artifacts/label_calibration.json")


def _file_digest(path: Path) -> str:
    sequence = load_landmark_sequence(path).landmarks
    return hashlib.sha256(sequence.tobytes()).hexdigest()


def _reference_catalog(reference_root: Path) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    kept: dict[str, list[str]] = {}
    exclusions: list[dict[str, str]] = []
    for label_dir in sorted(path for path in reference_root.iterdir() if path.is_dir()):
        seen_digests: set[str] = set()
        label_files: list[str] = []
        for path in sorted(label_dir.glob("*.npy")):
            try:
                sequence = load_landmark_sequence(path).landmarks
            except (OSError, ValueError) as error:
                exclusions.append({"path": str(path), "reason": str(error)})
                continue
            if np.all(sequence == 0):
                exclusions.append({"path": str(path), "reason": "all_zero"})
                continue
            digest = _file_digest(path)
            if digest in seen_digests:
                exclusions.append({"path": str(path), "reason": "exact_duplicate"})
                continue
            seen_digests.add(digest)
            label_files.append(path.name)
        kept[label_dir.name] = label_files
    return kept, exclusions


def _load_pair_info(pair_info_path: Path) -> np.ndarray:
    info = np.load(pair_info_path, allow_pickle=False)
    if info.ndim != 2 or info.shape[1] != 4:
        raise ValueError(
            f"{pair_info_path} must have columns [label, file_a, file_b, distance]."
        )
    return info


def _select_region_pair_samples(
    pairs: list[tuple[str, str, float]], max_samples: int = 24
) -> list[list[str]]:
    """Sample pairs across the stable 75% of each label's own distribution."""

    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda item: item[2])
    stable_count = max(1, int(np.ceil(len(pairs) * 0.75)))
    stable_pairs = pairs[:stable_count]
    indices = np.unique(
        np.linspace(0, len(stable_pairs) - 1, min(max_samples, len(stable_pairs))).astype(int)
    )
    return [[stable_pairs[index][0], stable_pairs[index][1]] for index in indices]


def build_calibration(
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
    pair_info_path: str | Path = DEFAULT_PAIR_INFO,
    output_path: str | Path = DEFAULT_OUTPUT,
    top_k: int = 3,
) -> dict[str, Any]:
    """Create leave-one-out, label-specific distance distributions.

    The existing pairwise DTW cache is used only after invalid and duplicate
    references are excluded. The scoring statistic is intentionally the same
    top-k mean used for a learner, avoiding the old apples-to-oranges baseline.
    """

    reference_root = Path(reference_root)
    pair_info_path = Path(pair_info_path)
    output_path = Path(output_path)
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_root}")
    if not pair_info_path.is_file():
        raise FileNotFoundError(f"Pairwise DTW cache not found: {pair_info_path}")

    catalog, exclusions = _reference_catalog(reference_root)
    info = _load_pair_info(pair_info_path)
    by_label: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    valid_pairs = 0
    for label, file_a, file_b, distance in info:
        label = str(label)
        if file_a not in catalog.get(label, ()) or file_b not in catalog.get(label, ()):
            continue
        by_label[label].append((str(file_a), str(file_b), float(distance)))
        valid_pairs += 1

    labels: dict[str, Any] = {}
    for label, files in catalog.items():
        neighbors: dict[str, list[float]] = {file: [] for file in files}
        pairs = by_label[label]
        for file_a, file_b, distance in pairs:
            neighbors[file_a].append(distance)
            neighbors[file_b].append(distance)

        loo_distances: list[float] = []
        for file in files:
            distances = sorted(neighbors[file])
            if distances:
                loo_distances.append(float(np.mean(distances[: min(top_k, len(distances))])))

        sample_count = len(files)
        labels[label] = {
            "reference_files": files,
            "reference_count": sample_count,
            "pair_count": len(pairs),
            "top_k": top_k,
            "loo_top_k_distances": sorted(loo_distances),
            "region_pair_samples": _select_region_pair_samples(pairs),
            "quality": "ok" if sample_count >= 10 else "low_sample",
        }

    payload: dict[str, Any] = {
        "format_version": 2,
        "reference_root": str(reference_root),
        "pair_info_path": str(pair_info_path),
        "top_k": top_k,
        "labels": labels,
        "exclusions": exclusions,
        "summary": {
            "label_count": len(labels),
            "reference_count": sum(len(files) for files in catalog.values()),
            "pair_count": valid_pairs,
            "excluded_count": len(exclusions),
            "low_sample_labels": sum(
                details["quality"] == "low_sample" for details in labels.values()
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sign-evaluation calibration data.")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--pair-info", default=str(DEFAULT_PAIR_INFO))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    payload = build_calibration(
        args.reference_root, args.pair_info, args.output, args.top_k
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
