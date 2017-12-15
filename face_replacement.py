import cv2
import numpy as np
import scipy.misc as sc
import scipy.sparse as sparse

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

    replacement_image = np.zeros(target.shape, dtype=np.uint8)

    replacement_faces_ims = [source[y:y+h, x:x+w] for (x,y,w,h) in source_faces]
    replacement_faces_ims = [resize(face, (h, w)) for face, (x,y, w,h)
                         in zip(replacement_faces_ims, target_faces)]

    for (x,y,w,h), face in zip(target_faces, replacement_faces_ims):
        face_im = (face * 255).astype(np.uint8)
        replacement_image[y:y+h, x:x+w, :] = face_im



def find_foreground(im, rect):
    (x, y, w, h) = rect
    mask = np.zeros(im.shape[:2])
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(im, mask, (x, y, x + w, y + h),
                bgdModel, fgdModel, 1, mode=cv2.GC_INIT_WITH_RECT)

    mask[(mask != cv2.GC_PR_FGD) & (mask != cv2.GC_FGD)] = 0
    mask = mask.astype(bool)

    return mask

