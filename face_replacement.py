import cv2
import numpy as np
import scipy.misc as sc
import scipy.sparse as sparse
from skimage import transform as tf
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

    bboxPolygon = np.array([[xR, yR], [xR + wR, yR], [xR + wR, yR + hR], [xR, yR + hR]])
    old = bboxPolygon

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
    for i, (source, target) in enumerate(zip(source_vid, target_vid)):
        # poly_mask = np.zeros(target.shape[:2])
        # cv2.fillConvexPoly(poly_mask, np.round(bboxPolygon), 4, 1)
        newFrame = resize(cv2.cvtColor(target[bboxPolygon[0,1]: bboxPolygon[2,1], bboxPolygon[0,0]: bboxPolygon[2,0]], cv2.COLOR_BGR2GRAY), (hR, wR))

        if i!=0:
            uint_oldFrame = (oldFrame * 255).astype(np.uint8)
            uint_newFrame = (newFrame * 255).astype(np.uint8)
            newPoints,  st, err = cv2.calcOpticalFlowPyrLK(uint_oldFrame, uint_newFrame, oldPoints, None)
            goodNew = newPoints[st == 1]
            goodOld = oldPoints[st == 1]

            newPoints = goodNew.reshape(-1,1,2)
            oldPoints = goodOld.reshape(-1, 1, 2)
            tform3 = tf.ProjectiveTransform()
            tform3.estimate(oldPoints[:,0, :], newPoints[:,0, :])
            matrix = tform3._inv_matrix

            bboxOut = forwardAffineTransform(matrix, np.array(bboxPolygon[:, 0], ndmin=2), np.array(bboxPolygon[:, 1], ndmin=2))
            # print out
            bboxPolygon = np.hstack([bboxOut[0], bboxOut[1]])
            # print bboxPolygon
            pts = np.round(bboxPolygon.reshape((-1, 1, 2))).astype(np.int32)

            '''SHOW THE BOUNDING BOX'''
            videoFrame = cv2.polylines(target, [pts], True, (0, 255, 255))
            plt.imshow(videoFrame)
            plt.show()

            '''SHOW THE FEATURE POINTS'''
            plt.imshow(newFrame)
            plt.scatter(newPoints[:, 0, 0], newPoints[:, 0, 1])
            plt.show()

            oldPoints = newPoints

            minX, minY = np.min(bboxPolygon[:,:], axis=0)
            maxX, maxY = np.max(bboxPolygon[:, :], axis=0)

            M = cv2.getPerspectiveTransform(old.astype(np.float32), bboxPolygon.astype(np.float32))

            frame2 = cv2.warpPerspective(target, M, (source.shape))

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

def forwardAffineTransform(T,v1,v2):
    v1 = np.transpose(v1)
    v2 = np.transpose(v2)

    vecSize = v1.shape[0]

    concVec = np.concatenate((v1,v2), axis=1)
    onesVec = np.ones((concVec.shape[0],1))

    U = np.concatenate((concVec,onesVec), axis=1)

    retMat = np.dot(U,T[:,0:2])

    return (retMat[:,0].reshape((vecSize,1)), retMat[:,1].reshape((vecSize,1)))

