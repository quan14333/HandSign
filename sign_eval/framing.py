"""Face/shoulder visibility checks and manual capture readiness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


SHOULDER_INDICES = (11, 12)
MIN_VISIBILITY = 0.60
COUNTDOWN_SECONDS = 3.0
# The border indicates readiness, not a required body size or position.
GUIDE_BOUNDS = (0.02, 0.02, 0.98, 0.98)


@dataclass(frozen=True)
class FramingResult:
    status: str
    message: str

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def assess_framing(landmarks: Sequence[Any], face_detected: bool) -> FramingResult:
    """Require a detected face and two visible, in-frame Pose shoulder points.

    Messages use ASCII Vietnamese because OpenCV's built-in font does not
    render Vietnamese diacritics. No inferred metric distance is displayed.
    """

    if not face_detected:
        return FramingResult("no_face", "Chua thay ro mat - huong mat ve camera")
    if len(landmarks) <= max(SHOULDER_INDICES):
        return FramingResult("no_shoulders", "Chua thay du hai vai - chinh lai goc camera")
    points = np.array([
        [landmarks[i].x, landmarks[i].y,
         getattr(landmarks[i], "visibility", None),
         getattr(landmarks[i], "presence", None)]
        for i in SHOULDER_INDICES
    ], dtype=float)
    if not np.isfinite(points).all() or np.any(points[:, 2:] < MIN_VISIBILITY):
        return FramingResult("not_visible", "Chua thay ro ca hai vai - chinh lai goc camera")
    if np.any(points[:, :2] < 0) or np.any(points[:, :2] > 1):
        return FramingResult("out_of_frame", "Hay dua ca hai vai vao trong hinh")
    return FramingResult("ready", "Da thay mat va hai vai - goc camera phu hop")


@dataclass
class CaptureGate:
    """Show readiness before S; only an explicit start request records video.

    Once recording starts, framing changes are advisory; never drop motion
    frames or restart the recording midway through a sign.
    """

    ready: bool = False
    recording: bool = False
    countdown_until: float | None = None

    def update(self, ready: bool, now: float) -> None:
        self.ready = ready
        if self.recording:
            return
        if not ready:
            self.countdown_until = None
        elif self.countdown_until is not None and now >= self.countdown_until:
            self.recording = True
            self.countdown_until = None

    def request_start(self, now: float) -> bool:
        """Start a countdown while green; never queue S or restart an active timer."""
        if not self.ready or self.recording or self.countdown_until is not None:
            return False
        self.countdown_until = now + COUNTDOWN_SECONDS
        return True
