"""Local HTTP adapter for the existing HandSign evaluation pipeline.
Run: python -m uvicorn api:app --host 127.0.0.1 --port 8001
"""
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from sign_eval.paths import CALIBRATION_PATH, SAMPLE_VIDEO_ROOT
from recognizer import ModelSetupError

app = FastAPI(title="HandSign practice")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv(
    "HANDSIGN_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(","), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
_evaluation_lock = Lock()
_sample_lock = Lock()
MAX_BYTES = 25 * 1024 * 1024
        

def lesson_labels():
    with CALIBRATION_PATH.open(encoding="utf-8") as source:
        return list(json.load(source)["labels"])
                                        

@app.get("/lessons")
def lessons():
    return lesson_labels()


@app.get("/samples/{label}")
def sample(label: str):
    if label not in lesson_labels():
        raise HTTPException(404, "Không tìm thấy động tác.")
    path = SAMPLE_VIDEO_ROOT / f"{label}.mp4"
    if not path.is_file():
        raise HTTPException(404, "Hiện không có video mẫu.")
    # Bundled FMP4 clips decode in OpenCV but not Chromium's HTML video player.
    # Cache VP8/WebM without modifying the original model demonstration assets.
    with _sample_lock:
        playable = browser_sample(path)
    return FileResponse(playable, media_type="video/webm")


def browser_sample(source: Path) -> Path:
    import cv2
    import math
    cache = source.parent.parent / "artifacts" / "browser_samples"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (source.stem + ".webm")
    if output.is_file() and output.stat().st_mtime >= source.stat().st_mtime:
        return output
    temporary = output.with_suffix(".partial.webm")
    capture = cv2.VideoCapture(str(source))
    writer = None
    count = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"VP80"), fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise HTTPException(503, "Chưa thể chuyển đổi video mẫu cho trình duyệt.")
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not count or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise HTTPException(503, "Không đọc được video mẫu.")
    temporary.replace(output)
    return output


def evaluate_video(label: str, video_path: Path, directory: Path):
    import cv2
    import mediapipe as mp
    import numpy as np
    from record import create_landmarkers, extract_hand_landmarks, extract_face_landmarks, trim_video
    from sign_eval.landmarks import trim_to_active_hands, zscore_normalize_sequence
    from evaluate import evaluate_recording

    frames, masks = [], []
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    mirrored_video = directory / "mirrored.mp4"
    try:
        if not capture.isOpened():
            raise ValueError("Không đọc được video. Hãy quay lại bằng Chrome hoặc Edge.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25
        with ExitStack() as stack:
            hand, face, _pose = create_landmarkers(stack)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if len(frames) >= min(1200, int(fps * 17)):
                    raise ValueError("Video quá dài. Mỗi lượt quay tối đa 15 giây.")
                # Desktop capture mirrors before both extraction and recognition.
                # MediaRecorder saves raw camera pixels, unlike the CSS preview.
                frame = cv2.flip(frame, 1)
                if writer is None:
                    writer = cv2.VideoWriter(str(mirrored_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot write normalized recording")
                writer.write(frame)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hands, valid = extract_hand_landmarks(hand.detect(image))
                facial, face_valid = extract_face_landmarks(face.detect(image))
                frames.append(np.concatenate([hands.reshape(-1, 2), facial], axis=0))
                masks.append([valid[0], valid[1], face_valid])
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if len(frames) < 2:
        raise ValueError("Video quá ngắn hoặc không đọc được. Hãy thực hiện trọn vẹn động tác.")
    validity = np.asarray(masks, dtype=bool)
    active = np.flatnonzero(validity[:, :2].any(axis=1))
    if len(active) < 2:
        return {"status": "not_scored", "form": {"score": None, "region_scores": {}},
                "feedback": [{"message": "Chưa ghi nhận rõ bàn tay. Hãy giữ tay trong khung hình và thực hiện trọn vẹn động tác."}]}
    trimmed, trimmed_validity = trim_to_active_hands(np.asarray(frames, dtype=np.float32), validity)
    sequence = directory / "record.npz"
    np.savez_compressed(sequence, landmarks=zscore_normalize_sequence(trimmed), validity=trimmed_validity)
    trimmed_video = directory / "trimmed.mp4"
    trim_video(mirrored_video, trimmed_video, int(active[0]), int(active[-1]))
    return evaluate_recording(label, sequence, trimmed_video, directory / "evaluation.json")


@app.post("/evaluate")
async def evaluate(label: str = Form(...), video: UploadFile = File(...)):
    if label not in lesson_labels():
        await video.close()
        raise HTTPException(422, "Động tác không có trong bộ mẫu HandSign.")
    if not _evaluation_lock.acquire(blocking=False):
        await video.close()
        raise HTTPException(503, "AI đang xử lý một lượt khác. Vui lòng thử lại sau.")
    try:
        with TemporaryDirectory(prefix="handsign-") as temporary:
            directory = Path(temporary)
            path = directory / ("capture.mp4" if video.content_type and "mp4" in video.content_type else "capture.webm")
            size = 0
            with path.open("wb") as target:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "Video quá lớn. Hãy quay một lượt ngắn hơn.")
                    target.write(chunk)
            if not size:
                raise HTTPException(422, "Video trống. Vui lòng quay lại.")
            return await run_in_threadpool(evaluate_video, label, path, directory)
    except ModelSetupError as error:
        logging.exception("HandSign model setup failed")
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logging.exception("HandSign evaluation failed")
        raise HTTPException(503, "Chưa thể chạy model HandSign. Kiểm tra môi trường Python và model, sau đó thử lại.") from error
    finally:
        await video.close()
        _evaluation_lock.release()
