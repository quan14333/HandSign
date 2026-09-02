
import numpy as np
from keras.models import load_model
import tensorflow as tf
import urllib.request
import mediapipe as mp
import cv2
url=[
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
]

path_hand="hand_landmarker.task"
path_face="face_landmarker.task"
def mediap_pipe(path, url):
    import os
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return mp.tasks.BaseOptions(model_asset_path=path)

baseoption_hand=mediap_pipe(path_hand,url[0])
baseoption_face=mediap_pipe(path_face,url[1])
options_hand=mp.tasks.vision.HandLandmarkerOptions(
    base_options=baseoption_hand,
    num_hands=2,    
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
options_face=mp.tasks.vision.FaceLandmarkerOptions(
    base_options=baseoption_face,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_faces=1
)
face_landmarker=mp.tasks.vision.FaceLandmarker.create_from_options(options_face)
hand_landmarker=mp.tasks.vision.HandLandmarker.create_from_options(options_hand)



timestamp=0
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Cannot open camera")
        break

    frame=cv2.flip(frame,1)
    h,w,_=frame.shape
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    hand_result=hand_landmarker.detect_for_video(mp_img,timestamp)
    face_result=face_landmarker.detect_for_video(mp_img,timestamp)
    timestamp+=25
    if hand_result.hand_landmarks or face_result.face_landmarks:
        for hand in hand_result.hand_landmarks:
            for landmark in hand:
                x=int(landmark.x*w)
                y=int(landmark.y*h)
                cv2.circle(frame,(x,y),5,(0,255,0),-1)
            connections=[
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),    
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]
            for start,end in connections:
                x1=int(hand[start].x*w)
                y1=int(hand[start].y*h)
                x2=int(hand[end].x*w)
                y2=int(hand[end].y*h)
                cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

            xs=[int(landmark.x*w) for landmark in hand]
            ys=[int(landmark.y*h) for landmark in hand]
            padding = 20

            x_min = max(min(xs) - padding*2, 0)
            x_max = min(max(xs) + padding*2, w)
            y_min = max(min(ys) - padding, 0)
            y_max = min(max(ys) + padding, h)

            cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,0,255),2)

        for face in face_result.face_landmarks:
            for landmark in face:
                x=int(landmark.x*frame.shape[1])
                y=int(landmark.y*frame.shape[0])
                cv2.circle(frame,(x,y),1,(255,255,255),-1)
                xs=[int(landmark.x*w) for landmark in face]
                ys=[int(landmark.y*h) for landmark in face]
                x_min = max(min(xs) -5, 0)
                x_max = min(max(xs)+5, w)
                y_min = max(min(ys)-20 , 0)
                y_max = min(max(ys) , h)
                cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(255,255,255),2)

    cv2.imshow('Hand Landmarker',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()