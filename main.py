
from keras.src.utils.audio_dataset_utils import tf
import numpy as np
from keras.models import load_model
import tensorflow as tf
import urllib.request
import mediapipe as mp
import cv2
url='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
urllib.request.urlretrieve(url,'hand_landmarker.task')
print('Done')

image_shape = (150,150, 3)
N_TYPES = 24

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    32, (4, 4),
    activation="relu",
    input_shape=image_shape
))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(tf.keras.layers.Conv2D(
    64, (4, 4),
    activation="relu"
))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(tf.keras.layers.Conv2D(
    128, (4, 4),
    activation="relu"
))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(tf.keras.layers.Conv2D(
    128, (4, 4),
    activation="relu"
))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(
    512,
    activation="relu"
))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(
    N_TYPES,
    activation="softmax"
))

model.load_weights(
    r"C:\Users\quan\Downloads\model.h5"
)

classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
     "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y"
]


baseoption=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
handLandmarkeroption=mp.tasks.vision.HandLandmarkerOptions
handLandmarker=mp.tasks.vision.HandLandmarker
runningNode=mp.tasks.vision.RunningMode

options=handLandmarkeroption(
    base_options=baseoption,
    num_hands=2,
    running_mode=runningNode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    min_hand_presence_confidence=0.5
)
handLandmarker=handLandmarker.create_from_options(options)
timestamp=0
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Cannot open camera")
        break
    frame=cv2.flip(frame,1)
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    result=handLandmarker.detect_for_video(mp_img,timestamp)
    timestamp+=25
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h,w,_=frame.shape
            for landmark in hand:
                x=int(landmark.x*w)
                y=int(landmark.y*h)
                cv2.circle(frame,(x,y),5,(0,255,0),-1)
            connections=[
                (0,1),(1,2),(2,3),(3,4),
                (1,5),(5,6),(6,7),(7,8),    
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
            crop=frame[y_min:y_max,x_min:x_max]
            crop=cv2.resize(crop,(150,150))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = crop.astype(np.float32)
            x = np.expand_dims(x, axis=0)
            x=x/255.0
            y_pred=model.predict(x)
            y_pred=np.argmax(y_pred)
            label=classes[y_pred]
            text=f"Predicted: {label}"+f'  confidence: {np.max(y_pred):.2f}'
            cv2.putText(frame, text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 214, 0), 2)
            

    cv2.imshow('Hand Landmarker',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()