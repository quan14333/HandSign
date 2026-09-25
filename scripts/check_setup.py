"""Check a fresh checkout without requiring optional demo videos or a camera."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures = []
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        failures.append("Use Python 3.10, 3.11 or 3.12.")
    for module in ("numpy", "fastdtw", "cv2", "mediapipe", "PIL", "torch", "transformers", "huggingface_hub", "tkinter"):
        if importlib.util.find_spec(module) is None:
            failures.append(f"Missing module: {module}. Install requirements.txt (Tkinter may need an OS package).")
    for name in ("hand_landmarker.task", "face_landmarker.task", "pose_landmarker_lite.task"):
        path = root / name
        if not path.is_file() or path.stat().st_size < 1024:
            failures.append(f"Missing model asset or Git LFS pointer: {name}")
    try:
        raw = (root / "artifacts/label_calibration.json").read_bytes()
        config = json.loads(raw)
        with zipfile.ZipFile(root / "artifacts/reference_landmarks.npz") as packed:
            if packed.comment != hashlib.sha256(raw).hexdigest().encode("ascii"):
                failures.append("Reference pack does not match calibration; regenerate it before sharing.")
            names = set(packed.namelist())
            missing = [f"{label}/{file}" for label, details in config["labels"].items()
                       for file in details["reference_files"] if f"{label}/{file}" not in names]
            if missing:
                failures.append(f"Reference pack is incomplete: {len(missing)} missing files.")
        sample_labels = {path.stem for path in (root / "sample_videos").glob("*.mp4")}
        expected_labels = set(config["labels"])
        if sample_labels != expected_labels:
            failures.append(
                "Bundled sample videos do not match the calibrated labels: "
                f"{len(expected_labels - sample_labels)} missing, "
                f"{len(sample_labels - expected_labels)} unexpected."
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        failures.append(f"Unable to read bundled references: {error}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    print("Setup files OK, including one bundled video for every label.")
    print("Internet is needed for the first AI model download.")
    print("Run record.py from this environment. A working desktop session and webcam are required for capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
