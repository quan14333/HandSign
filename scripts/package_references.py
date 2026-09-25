"""Build the small, video-free reference pack needed by fresh checkouts."""

import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    calibration = root / "artifacts/label_calibration.json"
    raw = calibration.read_bytes()
    config = json.loads(raw)
    source = Path(config["reference_root"].replace("\\", "/"))
    if not source.is_absolute():
        source = root / source
    destination = root / "artifacts/reference_landmarks.npz"
    temporary = destination.with_suffix(".npz.tmp")
    count = 0
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as packed:
            packed.comment = hashlib.sha256(raw).hexdigest().encode("ascii")
            for label, details in config["labels"].items():
                for filename in details["reference_files"]:
                    path = source / label / filename
                    packed.write(path, arcname=f"{label}/{filename}")
                    count += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Packed {count} references: {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
