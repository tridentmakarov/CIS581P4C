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

    replacement_faces_ims = [source[y:y+h, x:x+w] for (x, y, w, h) in source_faces]
    replacement_faces_ims = [resize(face, (hR, wR)) for face, (xR,yR, wR,hR)
                         in zip(replacement_faces_ims, target_faces)]

    gray = cv2.cvtColor(np.uint8(replacement_faces_ims[0]*255), cv2.COLOR_BGR2GRAY)
    oldPoints = cv2.goodFeaturesToTrack(gray, 50, 0.01, 8, mask=None, useHarrisDetector=False, blockSize=4, k=0.04)

    print oldPoints.shape
    # plt.imshow(np.uint8(replacement_faces_ims[0] * 255))
    # plt.scatter(oldPoints[:, 0, 0], oldPoints[:, 0, 1])
    #
    # plt.show()
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    for i, frame in enumerate(source_vid):

        newFrame = resize(cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY), (hR, wR))
        if i!=0:
            uint_oldFrame = (oldFrame * 255).astype(np.uint8)
            uint_newFrame = (newFrame * 255).astype(np.uint8)
            newPoints,  st, err = cv2.calcOpticalFlowPyrLK(uint_oldFrame, uint_newFrame, oldPoints, None)
            #newPoints,  st, err = cv2.calcOpticalFlowPyrLK(uint_oldFrame, uint_newFrame, oldPoints, None, **lk_params)
            goodNew = newPoints[st == 1]
            goodOld = oldPoints[st == 1]

            oldPoints = goodNew.reshape(-1,1,2)

            plt.imshow(newFrame)
            plt.scatter(oldPoints[:, 0, 0], oldPoints[:, 0, 1])
            plt.show()
        oldFrame = newFrame

    # for i, (x,y,w,h), face in zip(target_faces, replacement_faces_ims):
    #     face_im = (face * 255).astype(np.uint8)
    #     replacement_image[y:y+h, x:x+w, :] = face_im
    #
    #
    #     points = cv2.calcOpticalFlowPyrLK()




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

