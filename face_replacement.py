import cv2
import numpy as np
import scipy.misc as sc
import scipy.sparse as sparse
from skimage import transform as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from modified_poisson_blending import modified_poisson_blending as MPB
import math

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

    (x,y,w,h) = source_faces[0]
    #replacement_image = np.zeros(target.shape, dtype=np.uint8)

    #replacement_faces_ims_source = [source[y:y+h, x:x+w] for (x, y, w, h) in source_faces]
    replacement_face = source[y:y+h, x:x+w]
    replacement_faces_ims_target = [resize(replacement_face, (hR, wR)) for (xR,yR, wR,hR)
                         in target_faces]

    bboxPolygonsSource = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for (x, y, w, h) in source_faces]
    bboxPolygonsTarget = [np.array([[xR, yR], [xR + wR, yR], [xR + wR, yR + hR], [xR, yR + hR]]) for (xR,yR, wR,hR) in target_faces]
    bboxPolygonSource = bboxPolygonsSource[0]
    #bboxPolygonTarget = bboxPolygonsTarget[0]
    old_bboxes = bboxPolygonsTarget


    #graySource = cv2.cvtColor(np.uint8(replacement_faces_ims_source[0]*255), cv2.COLOR_BGR2GRAY)
    grayTarget = cv2.cvtColor(np.uint8(replacement_faces_ims_target[0] * 255), cv2.COLOR_BGR2GRAY)
    #oldPointsSource = cv2.goodFeaturesToTrack(graySource, 50, 0.01, 8, mask=None, useHarrisDetector=False, blockSize=4, k=0.04)
    oldPointsTarget = cv2.goodFeaturesToTrack(grayTarget, 50, 0.01, 8, mask=None, useHarrisDetector=False, blockSize=4, k=0.04)

    # print oldPointsTarget.shape
    # plt.imshow(np.uint8(replacement_faces_ims_target[0] * 255))
    # plt.scatter(oldPointsTarget[:, 0, 0], oldPointsTarget[:, 0, 1])

    plt.show()
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    for i, (source, target) in enumerate(zip(source_vid, target_vid)):

        newSources = [resize(cv2.cvtColor(target[bboxPolygonSource[0,1]: bboxPolygonSource[2,1], bboxPolygonSource[0,0]: bboxPolygonSource[2,0]], cv2.COLOR_BGR2GRAY), (hR, wR))
                      for (xR,yR, wR,hR) in target_faces]
        newTarget = target

        if i!=0:
            uint_oldTarget = (oldTarget * 255).astype(np.uint8)
            uint_newTarget = (newTarget * 255).astype(np.uint8)
            newPointsTarget,  st, err = cv2.calcOpticalFlowPyrLK(uint_oldTarget, uint_newTarget, oldPointsTarget, None)
            goodNew = newPointsTarget[st == 1]
            goodOld = oldPointsTarget[st == 1]

            newPointsTarget = goodNew.reshape(-1, 1, 2)
            oldPointsTarget = goodOld.reshape(-1, 1, 2)

            useable = np.where(np.sqrt((newPointsTarget[:, 0, 0] - oldPointsTarget[:, 0, 0]) ** 2 + (newPointsTarget[:, 0, 1] - oldPointsTarget[:, 0, 1]) ** 2) < 6)

            newWarpTarget = np.reshape(goodNew[useable[0], :], (-1, 2))
            oldWarpTarget = np.reshape(goodOld[useable[0], :], (-1, 2))

            tform3 = tf.ProjectiveTransform()
            tform3.estimate(oldWarpTarget[:, :], newWarpTarget[:, :])
            matrix = tform3._inv_matrix

            bboxesOut = [forwardAffineTransform(matrix, np.array(bboxPolygonTarget[:, 0], ndmin=2),
                                                np.array(bboxPolygonTarget[:, 1], ndmin=2))
                         for bboxPolygonTarget in bboxPolygonsTarget]
            # print out
            bboxPolygonsTarget = [np.hstack([bboxOut[0], bboxOut[1]]) for bboxOut in bboxesOut]
            # print bboxPolygonTarget
            pts = np.round(bboxPolygonTarget.reshape((-1, 1, 2))).astype(np.int32)

            '''SHOW THE BOUNDING BOX'''
            videoTarget = cv2.polylines(target, [pts], True, (0, 255, 255))
            #plt.imshow(videoTarget)
            #plt.show()

            '''SHOW THE FEATURE POINTS'''
            plt.imshow(newTarget)
            plt.scatter(newPointsTarget[:, 0, 0] + xR, newPointsTarget[:, 0, 1] + yR)
            plt.show()

            oldPointsTarget = newPointsTarget

            minX, minY = np.min(bboxPolygonTarget[:,:], axis=0)
            maxX, maxY = np.max(bboxPolygonTarget[:, :], axis=0)

            Ms = [cv2.getPerspectiveTransform(old.astype(np.float32), bboxPolygonTarget.astype(np.float32))
                 for old in old_bboxes]

            sourceWarps = [cv2.warpPerspective(source, M, source.shape[1::-1]) for M in Ms]
            sourceFaces = np.array([sourceWarp[y:y+h, x:x+w, :] for sourceWarp in sourceWarps])


            '''SHOW THE FEATURE POINTS'''
            for sourceF in sourceFaces:
                pass
                #plt.imshow(sourceF)
                #plt.show()



            modified_img = target.copy()
            #mask = find_foreground(source, source_faces[0])
            # if np.all(mask == 0):
            #     mask[:] = 1
            for (xR,yR, wR,hR), face in zip(target_faces, replacement_faces_ims_target):
                im_mask = np.ones(face.shape[:2], dtype=np.bool)
                #im_mask = resize(mask, face.shape[:2])
                #mask[bboxPolygonSource[0, 1]: bboxPolygonSource[2, 1],
                #bboxPolygonSource[0, 0]: bboxPolygonSource[2, 0]] = 0
                modified_img = MPB(face, target[yR:yR+hR, xR:xR+wR], im_mask, modified_img, xR, yR)

            '''SHOW THE FEATURE POINTS'''
            plt.imshow(modified_img)
            plt.show()

        oldTarget = newTarget
        print i

    # for i, (x,y,w,h), face in zip(target_faces, replacement_faces_ims):
    #     face_im = (face * 255).astype(np.uint8)
    #     replacement_image[y:y+h, x:x+w, :] = face_im
    #
    #
    #     points = cv2.calcOpticalFlowPyrLK()


def find_foreground_whole_im(im):
    rect = (0,0,im.shape[1], im.shape[0])
    return find_foreground(im, rect)

def find_foreground(im, rect):
    if im.dtype == np.float64:
        im_ = (im * 255).astype(np.uint8)
    elif im.dtype == np.uint8:
        im_ = im
    else:
        raise TypeError("im must have type np.uint8 or np.float64")
    (x, y, w, h) = rect
    mask = np.zeros(im_.shape[:2], dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    if x == 0 and y == 0 and w == im_.shape[1] and h == im_.shape[0]:
        mask[:] = cv2.GC_PR_FGD
        mask[mask.shape[0], mask.shape[1]] = cv2.GC_PR_BGD
        cv2.grabCut(im_, mask, None,
                    bgdModel, fgdModel, 2, mode=cv2.GC_INIT_WITH_MASK)
    else:
        cv2.grabCut(im_, mask, (x, y, x + w, y + h),
                    bgdModel, fgdModel, 1, mode=cv2.GC_INIT_WITH_RECT)

    mask[(mask != cv2.GC_PR_FGD) & (mask != cv2.GC_FGD)] = 0
    mask = mask.astype(bool)

    return mask

# From https://stackoverflow.com/questions/37363875/matlab-transformpointsforward-equivalent-in-python
def forwardAffineTransform(T,v1,v2):
    v1 = np.transpose(v1)
    v2 = np.transpose(v2)

    vecSize = v1.shape[0]

    concVec = np.concatenate((v1,v2), axis=1)
    onesVec = np.ones((concVec.shape[0],1))

    U = np.concatenate((concVec,onesVec), axis=1)

    retMat = np.dot(U,T[:,0:2])

    return (retMat[:,0].reshape((vecSize,1)), retMat[:,1].reshape((vecSize,1)))

