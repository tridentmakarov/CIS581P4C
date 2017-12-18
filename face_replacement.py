import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform as transform
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg
from modified_poisson_blending import modified_poisson_blending as MPB
from face_landmark import align_source_face_to_target


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def face_replacement(source_vid, target_vid, out_filename, filterImg):

    buf = 40
    buf2 = 0

    source = source_vid.get_data(0)
    target = target_vid.get_data(0)
    trackedVideo = imageio.get_writer(out_filename, fps=source_vid._meta['fps'])

    source_faces = detect_faces(source)
    target_faces = detect_faces(target)
    if source_faces.size == 0 or target_faces.size == 0:
        raise ValueError("Face could not be detected in source image")

    (x,y,w,h) = source_faces[0]
    (xR, yR, wR, hR) = target_faces[0]

    replacement_face = source[y-buf2:y+h+buf2, x-buf2:x+w+buf2]
    replacement_faces_ims_source = [resize(replacement_face, (hR, wR)) for (xR, yR, wR, hR)
                                    in target_faces]

    if np.any(filterImg):
        filterImg = resize(filterImg, [wR,hR, 4])

        # plt.imshow(filterImg)
        # plt.show()

    bboxPolygonsSource = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for (x, y, w, h) in source_faces]
    bboxTarget = [np.array([[xR, yR], [xR + wR, yR], [xR + wR, yR + hR], [xR, yR + hR]]) for (xR, yR, wR, hR) in target_faces]
    bboxSource = bboxPolygonsSource[0]
    #bboxPolygonTarget = bboxTarget[0]
    old_bboxes = bboxTarget


    # graySource = cv2.cvtColor(np.uint8(replacement_faces_ims_source[0]*255), cv2.COLOR_BGR2GRAY)
    grayTarget = cv2.cvtColor(np.uint8(target * 255), cv2.COLOR_BGR2GRAY)
    # oldPointsSource = cv2.goodFeaturesToTrack(graySource, 50, 0.01, 8, mask=None, useHarrisDetector=False, blockSize=4, k=0.04)
    targetMask = grayTarget[yR:yR+hR, xR:xR+wR]
    targetFeaturesOld = cv2.goodFeaturesToTrack(targetMask, 100, 0.01, 10, mask=None, useHarrisDetector=False, blockSize=4, k=0.04)

    # print targetFeaturesOld.shape
    # plt.imshow(np.uint8(target * 255))
    # plt.scatter(targetFeaturesOld[:, 0, 0]+xR, targetFeaturesOld[:, 0, 1]+yR)
    # plt.show()

    lk_params = dict(winSize=(100, 100),
                     maxLevel=15,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    oldTarget = target

    for i, (source, target) in enumerate(zip(source_vid, target_vid)):
    
        
        # plt.imshow(source[y - buf: y + w + buf, x - buf: x + w + buf])
        # plt.show()
        modified_img = align_source_face_to_target(source[y - buf: y + h + buf, x - buf: x + w + buf], target[yR - buf2: yR + hR + buf2, xR - buf2: xR + wR + buf2])
        
        plt.imshow(modified_img)
        plt.show()

        # [x - buf: x + w + buf, y - buf: y + w + buf]
        # [xR - buf: xR + wR + buf, yR - buf: yR + wR + buf]

        newTarget = cv2.cvtColor(np.uint8(target * 255), cv2.COLOR_BGR2GRAY)

        if i != 0:
            uint_oldTarget = oldTarget
            uint_newTarget = newTarget
            targetFeaturesNew,  st, err = cv2.calcOpticalFlowPyrLK(uint_oldTarget, uint_newTarget, targetFeaturesOld, None, **lk_params)

            # plt.imshow(newTarget)
            # plt.scatter(targetFeaturesNew[:,0, 0] + xR, targetFeaturesNew[:,0, 1] + yR)
            # plt.show()

            goodNew = targetFeaturesNew[st == 1]
            goodOld = targetFeaturesOld[st == 1]

            newPointsT = goodNew.reshape(-1, 1, 2)
            oldPointsT = goodOld.reshape(-1, 1, 2)

            useable = np.where(np.sqrt((newPointsT[:, 0, 0] - oldPointsT[:, 0, 0]) ** 2 + (newPointsT[:, 0, 1] - oldPointsT[:, 0, 1]) ** 2) < 4)

            newWarpTarget = np.reshape(goodNew[useable[0], :], (-1, 2))
            oldWarpTarget = np.reshape(goodOld[useable[0], :], (-1, 2))

            tform3 = transform.ProjectiveTransform()
            tform3.estimate(newPointsT[:, 0, :], oldPointsT[:, 0, :])
            matrix = tform3._inv_matrix



            bboxesOut = [forwardAffineTransform(matrix, np.array(bboxPolygonTarget[:, 0], ndmin=2),
                                                np.array(bboxPolygonTarget[:, 1], ndmin=2))
                         for bboxPolygonTarget in bboxTarget]
            # print out
            bboxT = [np.hstack([bboxOut[0], bboxOut[1]]) for bboxOut in bboxesOut]
            bboxTarget = np.array(bboxT)
            # print bboxPolygonTarget
            pts = np.round(bboxTarget.reshape((-1, 1, 2))).astype(np.int32)

            '''SHOW THE BOUNDING BOX'''
            videoTarget = cv2.polylines(target, [pts], True, (0, 255, 255))
            #plt.imshow(videoTarget)
            #plt.show()

            '''SHOW THE FEATURE POINTS'''
            # plt.imshow(newTarget)
            # plt.scatter(newPointsT[:, 0, 0] + xR, newPointsT[:, 0, 1] + yR)
            # plt.show()

            targetFeaturesOld = targetFeaturesNew

            Ms = [cv2.getPerspectiveTransform(old.astype(np.float32), bboxTarget.astype(np.float32))
                 for old in old_bboxes]

            sourceWarps = [cv2.warpPerspective(source, M, source.shape[1::-1]) for M in Ms]
            sourceFaces = np.array([sourceWarp[y:y+h, x:x+w, :] for sourceWarp in sourceWarps])


            '''SHOW THE FEATURE POINTS'''
            for sourceF in sourceFaces:
                pass
                #plt.imshow(sourceF)
                #plt.show()



            modified_img = target.copy()
            mask = find_foreground(source, (x,y,w,h))
            if np.all(mask == 0):
                print "bad mask"
                mask[:] = 1
            for (xR,yR, wR,hR), face in zip(target_faces, replacement_faces_ims_source):
                #im_mask = np.ones(face.shape[:2], dtype=np.bool)
                im_mask = resize(mask[y-buf:y+h+buf, x-buf:x+w+buf], face.shape[:2])

                modified_img = MPB(face, target[yR - buf:yR+hR + buf, xR - buf:xR+wR + buf], im_mask, modified_img, xR, yR)
                if np.any(filterImg):
                    curr_filterImg = (transform.warp(filterImg[:, :, :], tform3, output_shape=filterImg.shape[1::-1]))
                    curr_filterImg[:, :, 0:2] *= 255

                    for i in range(wR):
                        for j in range(hR):
                            modified_img[i + yR, j + xR, 0] = modified_img[i + yR, j + xR, 0] * (1-curr_filterImg[i, j, 3]) + curr_filterImg[i, j, 0] * (curr_filterImg[i, j, 3])
                            modified_img[i + yR, j + xR, 1] = modified_img[i + yR, j + xR, 1] * (1-curr_filterImg[i, j, 3]) + curr_filterImg[i, j, 1] * (curr_filterImg[i, j, 3])
                            modified_img[i + yR, j + xR, 2] = modified_img[i + yR, j + xR, 2] * (1-curr_filterImg[i, j, 3]) + curr_filterImg[i, j, 2] * (curr_filterImg[i, j, 3])



            '''SHOW FACE SWAPPED IMAGE'''
            fig = plt.figure()
            plt.imshow(modified_img)
            plt.show()
            #

            # # Creating video frame (this code was adapted from imageio.readthedocs.io)
            # canvas = plt.get_current_fig_manager().canvas
            # agg = canvas.switch_backends(FigureCanvasAgg)
            # agg.draw()
            # s = agg.tostring_rgb()
            # l, b, w, h = agg.figure.bbox.bounds
            # w, h = int(w), int(h)
            # buf = np.fromstring(s, dtype=np.uint8)
            # buf.shape = h, w, 3
            # trackedVideo.append_data(buf)
            # plt.close(fig)

        oldTarget = newTarget
        print "Frame", i

    # for i, (x,y,w,h), face in zip(target_faces, replacement_faces_ims):
    #     face_im = (face * 255).astype(np.uint8)
    #     replacement_image[y:y+h, x:x+w, :] = face_im
    #
    #
    #     points = cv2.calcOpticalFlowPyrLK()


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
        cv2.grabCut(im_, mask, (x, y, w, h),
                    bgdModel, fgdModel, 5, mode=cv2.GC_INIT_WITH_RECT)

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

    return retMat[:, 0].reshape((vecSize, 1)), retMat[:, 1].reshape((vecSize, 1))

