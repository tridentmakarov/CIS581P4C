import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def face_replacement(source_vid, target_vid):
    source = source_vid.get_data(0)
    target = target_vid.get_data(0)

    source_faces = detect_faces(source)
    target_faces = detect_faces(target)
    if source_faces.size == 0 or target_faces.size == 0:
        raise ValueError("Face could not be detected in source image")

    replacement_image = target.copy()


    # mask = np.zeros(replacement_image.shape)
    # for (x,y,w,h) in target_faces:
    #     replacement_image[y:y+h, x:x+w] = 0
    #     mask[y:y+h, x:x+w] = 1

    replacement_faces_ims = [source[y:y+h, x:x+w] for (x,y,w,h) in source_faces]
    replacement_faces_ims = [resize(face, (h, w)) for face, (x,y, w,h)
                         in zip(replacement_faces_ims, target_faces)]
    for (x,y,w,h), face in zip(target_faces, replacement_faces_ims):
        replacement_image[y:y+h, x:x+w, :] = (face * 255).astype(np.uint8)

    plt.imshow(replacement_image)
    plt.show()