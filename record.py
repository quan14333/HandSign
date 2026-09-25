"""Capture a practice recording and preserve landmark tracking quality."""

from __future__ import annotations

from contextlib import ExitStack
import argparse
import math
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from sign_eval.landmarks import trim_to_active_hands, zscore_normalize_sequence
from sign_eval.paths import PROJECT_ROOT, USER_ROOT
from sign_eval.framing import (
    GUIDE_BOUNDS, CaptureGate, FramingResult, assess_framing,
)


MAX_HANDS = 2
RIGHT_EYE_IDX = 33
LEFT_EYE_IDX = 263
NOSE_TIP_IDX = 1
MOUTH_IDX = 13
RIGHT_EAR_IDX = 234
LEFT_EAR_IDX = 454
FACE_LANDMARK_INDICES = [
    RIGHT_EYE_IDX,
    LEFT_EYE_IDX,
    NOSE_TIP_IDX,
    MOUTH_IDX,
    RIGHT_EAR_IDX,
    LEFT_EAR_IDX,
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
]


def create_landmarkers(stack: ExitStack) -> tuple[object, object, object]:
    base_options = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options(model_asset_path=str(PROJECT_ROOT / "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options(model_asset_path=str(PROJECT_ROOT / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options(model_asset_path=str(PROJECT_ROOT / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        stack.enter_context(vision.HandLandmarker.create_from_options(hand_options)),
        stack.enter_context(vision.FaceLandmarker.create_from_options(face_options)),
        stack.enter_context(vision.PoseLandmarker.create_from_options(pose_options)),
    )


def extract_hand_landmarks(result: object) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right landmark slots and their detector-validity mask."""

    hands = np.zeros((MAX_HANDS, 21, 2), dtype=np.float32)
    valid = np.zeros(MAX_HANDS, dtype=bool)
    for hand_index, hand in enumerate(result.hand_landmarks):
        if hand_index >= MAX_HANDS:
            break
        handedness = result.handedness[hand_index][0].category_name
        slot = 0 if handedness == "Left" else 1
        for landmark_index, landmark in enumerate(hand):
            hands[slot, landmark_index] = (landmark.x, landmark.y)
        valid[slot] = True
    return hands, valid


def extract_face_landmarks(result: object) -> tuple[np.ndarray, bool]:
    face_points = np.zeros((6, 2), dtype=np.float32)
    if not result.face_landmarks:
        return face_points, False
    landmarks = result.face_landmarks[0]
    for output_index, landmark_index in enumerate(FACE_LANDMARK_INDICES):
        landmark = landmarks[landmark_index]
        face_points[output_index] = (landmark.x, landmark.y)
    return face_points, True


def draw_landmarks(frame: np.ndarray, hand_result: object, face_result: object) -> None:
    for hand in hand_result.hand_landmarks:
        for landmark in hand:
            cv2.circle(
                frame,
                (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])),
                4,
                (0, 255, 0),
                -1,
            )
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                (int(hand[start].x * frame.shape[1]), int(hand[start].y * frame.shape[0])),
                (int(hand[end].x * frame.shape[1]), int(hand[end].y * frame.shape[0])),
                (255, 0, 0),
                2,
            )
    for face in face_result.face_landmarks:
        for landmark_index in FACE_LANDMARK_INDICES:
            landmark = face[landmark_index]
            cv2.circle(
                frame,
                (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])),
                3,
                (0, 0, 255),
                -1,
            )


def trim_video(video_path: Path, output_path: Path, start: int, end: int) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create video: {output_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Video is incomplete; please record the sign again.")
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def draw_capture_guide(
    frame: np.ndarray, framing: FramingResult, gate: CaptureGate, frame_count: int, now: float,
) -> np.ndarray:
    """Put single-pass text in separate panels outside the camera image."""

    height, width = frame.shape[:2]
    preview = cv2.copyMakeBorder(frame, 64, 78, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    color = (0, 200, 0) if framing.ready else (0, 180, 255)
    x_min, y_min, x_max, y_max = GUIDE_BOUNDS
    cv2.rectangle(preview, (int(x_min * width), 64 + int(y_min * height)),
                  (int(x_max * width), 64 + int(y_max * height)), color, 2)
    if gate.recording:
        status = f"DANG QUAY: {frame_count} frames | Q: dung"
    elif gate.countdown_until is not None:
        remaining = max(1, math.ceil(gate.countdown_until - now))
        status = f"Bat dau sau {remaining}... Giu mat va hai vai trong hinh"
        cv2.putText(preview, str(remaining), (width // 2 - 30, 64 + height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)
    elif framing.ready:
        status = "GOC CAMERA OK - Nhan S de quay | Q: thoat"
    else:
        status = "CAN CHINH GOC CAMERA - Cho khung xanh | Q: thoat"
    detail = "Bam vao cua so camera de dung phim S / Q"
    for text, y, text_color in [
        (status, 38, color), (framing.message, height + 94, (235, 235, 235)),
        (detail, height + 123, (180, 180, 180)),
    ]:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
        scale = min(0.65, max(0.1, (width - 24) / max(text_width, 1)))
        cv2.putText(preview, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, text_color, 1, cv2.LINE_AA)
    return preview


def record_sequence() -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Check framing, capture an uninterrupted sign, then trim and normalize."""

    raw_frames: list[np.ndarray] = []
    validity_frames: list[np.ndarray] = []
    gate = CaptureGate()
    print("Adjust the camera until your face and both shoulders are visible (green frame).")
    print("Press S when green for a 3-second countdown. Q finishes and evaluates.")
    with ExitStack() as stack:
        capture = cv2.VideoCapture(0)
        stack.callback(capture.release)
        stack.callback(cv2.destroyAllWindows)
        if not capture.isOpened():
            print("Unable to open webcam.")
            return None
        hand_landmarker, face_landmarker, pose_landmarker = create_landmarkers(stack)
        writer = None
        previous_timestamp_ms = -1
        started_at = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Unable to read a webcam frame.")
                break
            frame = cv2.flip(frame, 1)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = max(previous_timestamp_ms + 1,
                               int((time.perf_counter() - started_at) * 1000))
            previous_timestamp_ms = timestamp_ms
            pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
            face_result = face_landmarker.detect(image)
            face, face_validity = extract_face_landmarks(face_result)
            framing = assess_framing(
                pose_result.pose_landmarks[0] if pose_result.pose_landmarks else [],
                face_detected=face_validity,
            )
            now = time.perf_counter()
            gate.update(framing.ready, now)

            if gate.recording:
                if writer is None:
                    output_dir = USER_ROOT / "video_user"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output_dir / "original_video.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), 30,
                        (frame.shape[1], frame.shape[0]),
                    )
                    stack.callback(writer.release)
                    if not writer.isOpened():
                        raise RuntimeError("Unable to create the webcam recording file.")
                    print("Recording started.")
                hand_result = hand_landmarker.detect(image)
                hands, hand_validity = extract_hand_landmarks(hand_result)
                raw_frames.append(np.concatenate([hands.reshape(-1, 2), face], axis=0))
                validity_frames.append(
                    np.array([hand_validity[0], hand_validity[1], face_validity], dtype=bool)
                )
                writer.write(frame)
                draw_landmarks(frame, hand_result, face_result)
            preview = draw_capture_guide(frame, framing, gate, len(raw_frames), now)
            cv2.imshow("Sign Language Recorder", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                gate.request_start(now)
            elif key in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty("Sign Language Recorder", cv2.WND_PROP_VISIBLE) < 1:
                break

    if not raw_frames:
        print("No frames were recorded.")
        return None
    raw_landmarks = np.asarray(raw_frames, dtype=np.float32)
    validity = np.asarray(validity_frames, dtype=bool)
    try:
        trimmed_raw, trimmed_validity = trim_to_active_hands(raw_landmarks, validity)
    except ValueError as error:
        print(error)
        return None
    active_indices = np.flatnonzero(validity[:, :2].any(axis=1))
    start, end = int(active_indices[0]), int(active_indices[-1])
    if len(trimmed_raw) < 2:
        print("Recording is too short; perform the full sign and try again.")
        return None
    normalized = zscore_normalize_sequence(trimmed_raw)
    return normalized, trimmed_validity, start, end


def save_recording(recording: tuple[np.ndarray, np.ndarray, int, int]) -> tuple[Path, Path]:
    """Persist the matching landmarks/video before starting recognition."""
    sequence, validity, start, end = recording
    sequence_dir = USER_ROOT / "sequence_user"
    video_dir = USER_ROOT / "video_user"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    np.save(sequence_dir / "sequence.npy", sequence)
    np.savez_compressed(sequence_dir / "record.npz", landmarks=sequence, validity=validity)
    trim_video(video_dir / "original_video.mp4", video_dir / "trimmed_video.mp4", start, end)
    return sequence_dir / "record.npz", video_dir / "trimmed_video.mp4"


def main() -> None:
    from evaluate import evaluate_recording
    from sign_eval.desktop import choose_target_label, show_sample_window, show_evaluation_window, show_notice
    from sign_eval.evaluator import SignEvaluator, canonical_label

    parser = argparse.ArgumentParser(description="Record a sign and show AI evaluation in a desktop window.")
    parser.add_argument("--target-label", help="Target sign; omit to choose it in a window.")
    parser.add_argument(
        "--sample-video-root", type=Path,
        help="Optional folder containing <label>.mp4 files or one subfolder per label.",
    )
    args = parser.parse_args()
    try:
        labels = SignEvaluator().available_labels()
        if args.target_label:
            lookup = {canonical_label(label): label for label in labels}
            target_label = lookup.get(canonical_label(args.target_label))
            if target_label is None:
                raise ValueError(f"Không tìm thấy từ mục tiêu: {args.target_label}")
        else:
            target_label = choose_target_label(labels)
    except Exception as error:
        show_notice("Không thể bắt đầu", str(error), error=True)
        return
    while target_label is not None:
        sample_action = show_sample_window(target_label, args.sample_video_root)
        if sample_action == "choose_another":
            target_label = choose_target_label(labels)
            continue
        if not sample_action:
            return
        break
    if target_label is None:
        return

    while True:
        try:
            recording = record_sequence()
        except Exception as error:
            show_notice("Không thể quay video", str(error), error=True)
            return
        if recording is None:
            show_notice("Chưa có video để chấm", "Lần quay chưa có đủ dữ liệu tay hợp lệ. Chưa chạy đánh giá.")
            return

        def process() -> dict:
            sequence_path, video_path = save_recording(recording)
            return evaluate_recording(target_label, sequence_path, video_path)

        if not show_evaluation_window(process, target_label):
            break


if __name__ == "__main__":
    main()
