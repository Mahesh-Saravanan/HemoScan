import numpy as np
from PIL import Image
import mediapipe as mp
import os
import cv2
def detect_hands(image):


    if isinstance(image, str) and os.path.isfile(image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.shape[-1] == 4: 
        image = image[:, :, :3]
    
 
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        results = hands.process(image)

        if results.multi_hand_landmarks:
            return {"status": True, "count": len(results.multi_hand_landmarks)}
        else:
            return {"status": False, "count": None}
        
file = "/Users/maheshsaravanan/Desktop/PF.png"
print(detect_hands(file))