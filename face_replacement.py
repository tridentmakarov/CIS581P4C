import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform as transform
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg
from modified_poisson_blending import modified_poisson_blending as MPB
from face_landmark import align_source_face_to_target, get_face_landmarks


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def face_replacement(source_vid, target_vid, out_filename, filterImg, debug=False):

    source = source_vid.get_data(0)
    target = target_vid.get_data(0)

    s_fps = source_vid._meta['fps']
    t_fps = target_vid._meta['fps']

    trackedVideo = imageio.get_writer(out_filename, fps=s_fps)

    lk_params = dict(winSize=(100, 100),
                     maxLevel=15,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    oldTarget = target
    j=0

    old_source_landmarks = []

    for i, (source, target) in enumerate(zip(source_vid, target_vid)):



        source_landmarks, source_locations = get_face_landmarks(source)
        target_landmarks, target_locations = get_face_landmarks(target)
        if (len(source_landmarks) == 0 or len(target_landmarks) == 0) and j == 0:
            print "no faces found, skipping"
        else:
            j += 1

            target_landmarks_flow, st, err = cv2.calcOpticalFlowPyrLK(oldTarget, target, target_landmarks, None,
                                                                  **lk_params)

            if len(source_landmarks) == 0 and j!=1:
                source_landmarks = old_source_landmarks

            if len(target_landmarks) == 0 and j!=1:



            warped_source, mask = align_source_face_to_target(source, target, source_landmarks, target_landmarks)
            modified_img = MPB(warped_source, None, mask, target)
            oldTarget = target

            # newTarget = cv2.cvtColor(np.uint8(target * 255), cv2.COLOR_BGR2GRAY)





            #     uint_oldTarget = oldTarget
            #     uint_newTarget = newTarget

            #
            #     goodNew = targetFeaturesNew[st == 1]
            #     goodOld = targetFeaturesOld[st == 1]
            #
            #     newPointsT = goodNew.reshape(-1, 1, 2)
            #     oldPointsT = goodOld.reshape(-1, 1, 2)
            #
            #     useable = np.where(np.sqrt((newPointsT[:, 0, 0] - oldPointsT[:, 0, 0]) ** 2 + (newPointsT[:, 0, 1] - oldPointsT[:, 0, 1]) ** 2) < 4)
            #
            #     tform3 = transform.ProjectiveTransform()
            #     tform3.estimate(newPointsT[:, 0, :], oldPointsT[:, 0, :])
            #     matrix = tform3._inv_matrix
            #
            #
            #
            #     '''SHOW THE BOUNDING BOX'''
            #
            #     if debug:
            #         bboxesOut = [forwardAffineTransform(matrix, np.array(bboxPolygonTarget[:, 0], ndmin=2),
            #                                             np.array(bboxPolygonTarget[:, 1], ndmin=2))
            #                      for bboxPolygonTarget in bboxTarget]
            #
            #         bboxT = [np.hstack([bboxOut[0], bboxOut[1]]) for bboxOut in bboxesOut]
            #         bboxTarget = np.array(bboxT)
            #         print bboxPolygonTarget
            #         pts = np.round(bboxTarget.reshape((-1, 1, 2))).astype(np.int32)
            #
            #         videoTarget = cv2.polylines(target, [pts], True, (0, 255, 255))
            #         plt.imshow(videoTarget)
            #         plt.show()
            #
            #         '''SHOW THE FEATURE POINTS'''
            #         plt.imshow(newTarget)
            #         plt.scatter(newPointsT[:, 0, 0] + xR, newPointsT[:, 0, 1] + yR)
            #         plt.show()
            #
            #     targetFeaturesOld = targetFeaturesNew
            #
            #     Ms = [cv2.getPerspectiveTransform(old.astype(np.float32), bboxTarget.astype(np.float32))
            #          for old in old_bboxes]
            #
            #     sourceWarps = [cv2.warpPerspective(source, M, source.shape[1::-1]) for M in Ms]
            #     sourceFaces = np.array([sourceWarp[y:y+h, x:x+w, :] for sourceWarp in sourceWarps])
            #
            #
            #     '''SHOW THE FEATURE POINTS'''
            #     for sourceF in sourceFaces:
            #         pass
            #         #plt.imshow(sourceF)
            #         #plt.show()
            #
            #         warped_source, mask = align_source_face_to_target(source, target)
            #         modified_img = MPB(warped_source, None, mask, target)
            #         plt.imshow(modified_img)
            #         plt.show()
            #
            #         # modified_img = MPB(face, target[yR - buf:yR+hR + buf, xR - buf:xR+wR + buf], im_mask, modified_img, xR, yR)
            #         # if np.any(filterImg):
            #         #     curr_filterImg = (transform.warp(filterImg[:, :, :], tform3, output_shape=filterImg.shape[1::-1]))
            #         #     curr_filterImg[:, :, 0:2] *= 255
            #         #
            #         #     for i in range(wR):
            #         #         for j in range(hR):
            #         #             modified_img[i + yR, j + xR, 0] = modified_img[i + yR, j + xR, 0] * (1-curr_filterImg[i, j, 3]) + curr_filterImg[i, j, 0] * (curr_filterImg[i, j, 3])
            #         #             modified_img[i + yR, j + xR, 1] = modified_img[i + yR, j + xR, 1] * (1-curr_filterImg[i, j, 3]) + curr_filterImg[i, j, 1] * (curr_filterImg[i, j, 3])
            #         #             modified_img[i + yR, j + xR, 2] = modified_img[i + yR, j + xR, 2] * (1-curr_filterImg[i, j, 3]) + curr_filterImg[i, j, 2] * (curr_filterImg[i, j, 3])
            #         #


            '''SHOW FACE SWAPPED IMAGE'''
            fig = plt.figure()
            plt.imshow(modified_img)
            plt.show()
            #

            # # Creating video frame (this code was adapted from imageio.readthedocs.io)
            '''
            canvas = plt.get_current_fig_manager().canvas
            agg = canvas.switch_backends(FigureCanvasAgg)
            agg.draw()
            s = agg.tostring_rgb()
            l, b, w, h = agg.figure.bbox.bounds
            w, h = int(w), int(h)
            buf = np.fromstring(s, dtype=np.uint8)
            buf.shape = h, w, 3
            trackedVideo.append_data(buf)
            plt.close(fig)
            '''
            print "Frame", j


            old_source_landmarks = source_landmarks
            old_target_landmarks = target_landmarks




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

