import cv2
import numpy as np
import scipy.misc as sc
import scipy.sparse as sparse
from skimage import transform as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from modified_poisson_blending import modified_poisson_blending as MPB

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

    replacement_faces_ims_source = [source[y:y+h, x:x+w] for (x, y, w, h) in source_faces]
    replacement_faces_ims_target = [resize(face, (hR, wR)) for face, (xR,yR, wR,hR)
                         in zip(replacement_faces_ims_source, target_faces)]

    bboxPolygonsSource = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for (x, y, w, h) in source_faces]
    bboxPolygonsTarget = [np.array([[xR, yR], [xR + wR, yR], [xR + wR, yR + hR], [xR, yR + hR]]) for (xR,yR, wR,hR) in target_faces]
    bboxPolygonSource = bboxPolygonsSource[0]
    bboxPolygonTarget = bboxPolygonsTarget[0]
    old = bboxPolygonTarget

    graySource = cv2.cvtColor(np.uint8(replacement_faces_ims_source[0]*255), cv2.COLOR_BGR2GRAY)
    grayTarget = cv2.cvtColor(np.uint8(replacement_faces_ims_target[0] * 255), cv2.COLOR_BGR2GRAY)
    oldPointsSource = cv2.goodFeaturesToTrack(graySource, 50, 0.01, 8, mask=None, useHarrisDetector=False, blockSize=4, k=0.04)
    oldPointsTarget = cv2.goodFeaturesToTrack(grayTarget, 50, 0.01, 8, mask=None, useHarrisDetector=False, blockSize=4, k=0.04)

    print oldPointsTarget.shape
    # plt.imshow(np.uint8(replacement_faces_ims[0] * 255))
    # plt.scatter(oldPoints[:, 0, 0], oldPoints[:, 0, 1])
    #
    # plt.show()
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    for i, (source, target) in enumerate(zip(source_vid, target_vid)):

        newSource = resize(cv2.cvtColor(target[bboxPolygonSource[0,1]: bboxPolygonSource[2,1], bboxPolygonSource[0,0]: bboxPolygonSource[2,0]], cv2.COLOR_BGR2GRAY), (hR, wR))
        newTarget = target

        if i!=0:
            uint_oldTarget = (oldTarget * 255).astype(np.uint8)
            uint_newTarget = (newTarget * 255).astype(np.uint8)
            newPointsTarget,  st, err = cv2.calcOpticalFlowPyrLK(uint_oldTarget, uint_newTarget, oldPointsTarget, None)
            goodNew = newPointsTarget[st == 1]
            goodOld = oldPointsTarget[st == 1]

            newPointsTarget = goodNew.reshape(-1, 1, 2)
            oldPointsTarget = goodOld.reshape(-1, 1, 2)
            tform3 = tf.ProjectiveTransform()
            tform3.estimate(oldPointsTarget[:,0, :], newPointsTarget[:,0, :])
            matrix = tform3._inv_matrix

            bboxOut = forwardAffineTransform(matrix, np.array(bboxPolygonTarget[:, 0], ndmin=2), np.array(bboxPolygonTarget[:, 1], ndmin=2))
            # print out
            bboxPolygonTarget = np.hstack([bboxOut[0], bboxOut[1]])
            # print bboxPolygonTarget
            pts = np.round(bboxPolygonTarget.reshape((-1, 1, 2))).astype(np.int32)

            '''SHOW THE BOUNDING BOX'''
            videoTarget = cv2.polylines(target, [pts], True, (0, 255, 255))
            plt.imshow(videoTarget)
            plt.show()

            '''SHOW THE FEATURE POINTS'''
            plt.imshow(newTarget)
            plt.scatter(newPointsTarget[:, 0, 0], newPointsTarget[:, 0, 1])
            plt.show()

            oldPointsTarget = newPointsTarget

            minX, minY = np.min(bboxPolygonTarget[:,:], axis=0)
            maxX, maxY = np.max(bboxPolygonTarget[:, :], axis=0)

            M = cv2.getPerspectiveTransform(old.astype(np.float32), bboxPolygonTarget.astype(np.float32))

            sourceWarp = cv2.warpPerspective(source, M, source.shape[1::-1])

            '''SHOW THE FEATURE POINTS'''
            plt.imshow(sourceWarp)
            plt.show()

            mask = np.ones(newSource.shape)
            mask[bboxPolygonSource[0,1]: bboxPolygonSource[2,1], bboxPolygonSource[0,0]: bboxPolygonSource[2,0]] = 0

            modified_img = MPB(sourceWarp, target, bboxPolygonSource, target, x, y)

            '''SHOW THE FEATURE POINTS'''
            plt.imshow(modified_img)
            plt.show()

        oldTarget = newTarget

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

