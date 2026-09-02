import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import time



url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

urllib.request.urlretrieve(
    url,
    "hand_landmarker.task"
)



base_option = mp.tasks.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_option,
    num_hands=2,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

handmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)



def extract_landmarks(result):

    hands = np.zeros(
        (2, 21, 3),
        dtype=np.float32
    )

    for i, hand in enumerate(result.hand_landmarks):

        if i >= 2:
            break

        for j, landmark in enumerate(hand):

            hands[i, j] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

    return hands.flatten()



def record_sequence():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được webcam.")
        return None

    sequence = []
    recording = False
    countdown = False

    start_time = None
    countdown_start = None

    last_timestamp = -1

    print("--------------------------------")
    print("Nhấn S để bắt đầu")
    print("Nhấn Q để thoát")
    print("--------------------------------")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Không đọc được frame.")
            break

        frame = cv2.flip(frame, 1)


        if countdown:

            elapsed = time.perf_counter() - countdown_start

            remaining = 3 - int(elapsed)

            if remaining > 0:

                cv2.putText(
                    frame,
                    str(remaining),
                    (frame.shape[1] // 2 - 40,
                     frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (0, 255, 255),
                    8
                )

                cv2.putText(
                    frame,
                    "Chuan bi...",
                    (frame.shape[1] // 2 - 120,
                     frame.shape[0] // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            else:

                countdown = False
                recording = True

                sequence = []

                start_time = time.perf_counter()

                last_timestamp = -1

                print(">>> BẮT ĐẦU GHI!")


        if recording:


            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1

            last_timestamp = timestamp_ms


            result = handmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )


            for hand in result.hand_landmarks:

                for landmark in hand:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )


            landmarks = extract_landmarks(result)

            sequence.append(landmarks)


            cv2.putText(
                frame,
                f"RECORDING: {len(sequence)} frames",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        elif not countdown:

            cv2.putText(
                frame,
                "Press S to start",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        cv2.imshow(
            "Sign Language Recorder",
            frame
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):

            if not recording and not countdown:

                countdown = True

                countdown_start = time.perf_counter()

                print(">>> Chuẩn bị...")

        elif key == ord("q"):

            if recording or countdown:

                print(">>> Dừng ghi.")

                break

            else:

                break
    cap.release()
    cv2.destroyAllWindows()
    if len(sequence) == 0:
        print("Không có frame nào được ghi.")
        return None
    sequence = np.array(
        sequence,
        dtype=np.float32
    )
    print("Đã ghi xong!")
    print("Shape:", sequence.shape)
    return sequence
sequence = record_sequence()
if sequence is not None:

    np.save(
        "webcam_sequence.npy",
        sequence
    )

    print("Đã lưu: webcam_sequence.npy")