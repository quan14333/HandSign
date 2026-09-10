"""Capture a practice recording and preserve landmark tracking quality."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from sign_eval.landmarks import trim_to_active_hands, zscore_normalize_sequence


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


def create_landmarkers() -> tuple[object, object]:
    base_options = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return (
        vision.HandLandmarker.create_from_options(hand_options),
        vision.FaceLandmarker.create_from_options(face_options),
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
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
    capture.release()
    writer.release()


def record_sequence() -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Capture landmarks, trim to the active signing portion, and normalize it."""

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Unable to open webcam.")
        return None
    output_dir = Path("data/user/video_user")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_video_path = output_dir / "original_video.mp4"
    writer = cv2.VideoWriter(
        str(original_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    hand_landmarker, face_landmarker = create_landmarkers()
    raw_frames: list[np.ndarray] = []
    validity_frames: list[np.ndarray] = []
    recording = False
    countdown_started_at: float | None = None
    print("Press S to start recording and Q to stop.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Unable to read a webcam frame.")
                break
            frame = cv2.flip(frame, 1)
            if countdown_started_at is not None:
                remaining = 3 - int(time.perf_counter() - countdown_started_at)
                if remaining > 0:
                    cv2.putText(
                        frame, str(remaining), (frame.shape[1] // 2 - 40, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8,
                    )
                else:
                    countdown_started_at = None
                    recording = True
                    raw_frames.clear()
                    validity_frames.clear()
                    print("Recording started.")

            if recording:
                writer.write(frame)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                hand_result = hand_landmarker.detect(image)
                face_result = face_landmarker.detect(image)
                hands, hand_validity = extract_hand_landmarks(hand_result)
                face, face_validity = extract_face_landmarks(face_result)
                raw_frames.append(np.concatenate([hands.reshape(-1, 2), face], axis=0))
                validity_frames.append(
                    np.array([hand_validity[0], hand_validity[1], face_validity], dtype=bool)
                )
                draw_landmarks(frame, hand_result, face_result)
                cv2.putText(
                    frame, f"RECORDING: {len(raw_frames)} frames", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                )
            elif countdown_started_at is None:
                cv2.putText(
                    frame, "Press S to start", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            cv2.imshow("Sign Language Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and not recording and countdown_started_at is None:
                countdown_started_at = time.perf_counter()
            elif key == ord("q"):
                break
    finally:
        writer.release()
        capture.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()

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
    normalized = zscore_normalize_sequence(trimmed_raw)
    return normalized, trimmed_validity, start, end


def main() -> None:
    recording = record_sequence()
    if recording is None:
        return
    sequence, validity, start, end = recording
    sequence_dir = Path("data/user/sequence_user")
    video_dir = Path("data/user/video_user")
    sequence_dir.mkdir(parents=True, exist_ok=True)
    np.save(sequence_dir / "sequence.npy", sequence)
    np.savez_compressed(sequence_dir / "record.npz", landmarks=sequence, validity=validity)
    trim_video(video_dir / "original_video.mp4", video_dir / "trimmed_video.mp4", start, end)
    print(f"Saved {len(sequence)} normalized frames to {sequence_dir / 'record.npz'}.")


if __name__ == "__main__":
    main()
