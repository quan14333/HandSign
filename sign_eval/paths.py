"""Default assets are relative to the checkout, never a developer's CWD."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "label_calibration.json"
REFERENCE_ARCHIVE = PROJECT_ROOT / "artifacts" / "reference_landmarks.npz"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "landmarks (1)"
USER_ROOT = PROJECT_ROOT / "data" / "user"
SAMPLE_VIDEO_ROOT = PROJECT_ROOT / "sample_videos"


def resolve_project_path(value: str | Path) -> Path:
    """Accept legacy Windows separators in saved calibration on other OSes."""
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path
