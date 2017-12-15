import cv2
import numpy as np
import scipy.misc as sc
import scipy.sparse as sparse

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

def modified_poisson_blending(source, target, mask, originalTarget, x_corner, y_corner):
    
    source /= 255
    target /= 255
    mask /= 255
    
    F = np.zeros(source.shape)
    
    F = 1-F
    
    
    
    return output_img

# Poisson blending using http://vacation.aid.design.kyushu-u.ac.jp/and/poisson/
def poisson_gray(source, target, mask):
    n = source.size
    A = sparse(n, n)
    f = np.zeros(n, 1)
    fx = np.where(mask > 0)
    bx = np.where(mask == 0)
    q = np.zeros(n,1)
    q[fx] = 1
    
    I = np.diag(sparse(q))
    A = -4*I
    
    
    
    return F