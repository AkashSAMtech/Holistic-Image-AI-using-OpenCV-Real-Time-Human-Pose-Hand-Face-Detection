from mss import mss
import cv2
from PIL import Image
import numpy as np
from time import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


mon = {'top': 100, 'left': 200, 'width': 1600, 'height': 1024}
sct = mss()

while True:
    begin_time = time()


    sct_img = sct.grab(mon)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


    results = holistic.process(img_rgb)


    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(img_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(img_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(img_bgr, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)


    cv2.imshow('Holistic Detection', img_bgr)
    print('This frame takes {:.3f} seconds.'.format(time() - begin_time))


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
holistic.close()
