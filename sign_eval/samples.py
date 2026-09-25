"""Optional local demonstration videos, independent of scoring references."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import cv2

from .evaluator import canonical_label
from .paths import SAMPLE_VIDEO_ROOT

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sample_candidates(label: str, root: Path | None = None) -> list[Path]:
    configured = root or os.environ.get("HANDSIGN_SAMPLE_VIDEO_ROOT") or SAMPLE_VIDEO_ROOT
    folder = Path(configured).expanduser()
    try:
        children = sorted(folder.iterdir())
        candidates = []
        for child in children:
            if child.is_dir() and canonical_label(child.name) == canonical_label(label):
                candidates.extend(path for path in sorted(child.iterdir())
                                  if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
            elif child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES and canonical_label(child.stem) == canonical_label(label):
                candidates.append(child)
        return candidates
    except OSError:
        return []


@dataclass
class SampleVideo:
    capture: Any
    first_frame: Any
    fps: float

    def close(self) -> None:
        self.capture.release()


def open_sample_video(label: str, root: Path | None = None) -> SampleVideo | None:
    """Only offer playback after both open and first-frame decoding succeed."""
    for path in sample_candidates(label, root):
        capture = None
        try:
            capture = cv2.VideoCapture(str(path))
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    if not math.isfinite(fps) or fps <= 0:
                        fps = 25.0
                    return SampleVideo(capture, frame, fps)
        except (OSError, ValueError, cv2.error):
            pass
        if capture is not None:
            capture.release()
    return None
