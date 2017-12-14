import cv2
import numpy as np
def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def face_replacement(source_vid, target_vid):
    forground = source_vid.get_data(0)
    background = target_vid.get_data(0)

    faces = detect_faces(forground)
    if faces.size == 0:
        raise ValueError("Face could not be detected in source image")

    replacement_image = forground.copy()
    mask = np.zeros(replacement_image.shape)
    for (x,y,w,h) in faces:
        replacement_image[y:y+h, x:x+w] = 0
        mask[y:y+h, x:x+w] = 1

