CLASSES = ['fist', 'like', 'ok', 'one', 'palm', 'peace', 'rock', 'stop']

from array import array
import mediapipe as mp
import cv2
import os
import numpy as np
from functools import reduce
import math
from tensorflow import keras
from scipy.spatial import distance_matrix

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
def extractLandmarks(frame):

    landmarks = []
    # frame = cv2.imread(folder + '/' + img)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands_results = hands.process(frame)

    if not hands_results.multi_hand_landmarks:
      return

    # get hands result
    left_hand = [0] * 63
    right_hand = [0] * 63
    handedness = []

    hand_landmarks = hands_results.multi_hand_landmarks[0]
    landmarks = reduce(lambda x, lm: x + [lm.x, lm.y, lm.z], hand_landmarks.landmark, [])

    return landmarks


myModelKeras = keras.models.load_model('weights/static_model_1_1')
print(myModelKeras.summary())

def returnClass(frame):
    global myModelKeras
    landmarks = extractLandmarks(frame)
    
    if landmarks == None:
        return None
    landmarks = np.array(landmarks).reshape(21, 3)

    adjency = distance_matrix(landmarks,landmarks)
    return CLASSES[myModelKeras.predict(np.array([adjency])).argmax()]

if __name__=="main":
    pass