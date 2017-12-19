import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform as transform
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg
from modified_poisson_blending import modified_poisson_blending as MPB
from face_landmark import align_source_face_to_target, get_face_landmarks

#https://matthewearl.github.io/2015/07/28/switching-eds-with-python/

def to_gray(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def face_replacement(source_vid, target_vid, out_filename, filter_im, debug=False):

    source = source_vid.get_next_data()
    target = target_vid.get_next_data()

    s_fps = source_vid._meta['fps']
    t_fps = target_vid._meta['fps']

    trackedVideo = imageio.get_writer(out_filename, fps=s_fps)

    lk_params = dict(winSize=(100, 100),
                     maxLevel=15,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_target = target
    old_gray = to_gray(target)

    j=0

    old_source_landmarks = []
    old_target_landmarks = []

    points_list = []
    for i, (source, target) in enumerate(zip(source_vid, target_vid)):



        source_landmarks, source_locations = get_face_landmarks(source)
        target_landmarks, target_locations = get_face_landmarks(target)

        gray = to_gray(target)

        #Won't hit IndexError because i % 25 == 0 will trigger on first iteration.
        if i % 25 == 0 or any([np.all(~points['good_points']) for points in points_list[-1]]):
            current_points = [{"points":landmark, "good_points":np.ones(landmark.shape[0], dtype=bool)}
                              for landmark in target_landmarks]
        else:
            current_points = []
            for old_point in points_list[-1]:
                new_pts = np.zeros_like(old_point["points"], dtype=np.float32)
                new_state = old_point["good_points"].copy()
                new_pts[old_point["good_points"]], st, err = \
                    cv2.calcOpticalFlowPyrLK(old_gray, gray, old_point["points"][old_point["good_points"]].astype(np.float32)
                                             , None, **lk_params)
                new_state[old_point["good_points"]] = np.squeeze(st) == 1
                current_points.append({"points":new_pts, "good_points": new_state})
        points_list.append(current_points)

        if (len(source_landmarks) == 0 or len(target_landmarks) == 0) and j == 0:
            print "no faces found, skipping"
        else:
            j += 1

            # target_landmarks_flow, st, err = cv2.calcOpticalFlowPyrLK(oldTarget, target, np.array(target_landmarks), None,
            #                                                       **lk_params)

            if len(source_landmarks) == 0:
                source_landmarks = old_source_landmarks

            if len(target_landmarks) == 0:
                target_landmarks = old_target_landmarks


            warped_source, mask = align_source_face_to_target(source, target, source_landmarks, target_landmarks, tracked_points=None)
            modified_img = MPB(warped_source, None, mask, target)

            if np.any(filter_im):
                filter_im = np.array(filter_im*255).astype(np.uint8)
                w, h = filter_im.shape[0], filter_im.shape[1]
                filter_area = np.array([[0,0],[w, 0],[w, h], [0,w]]).astype(np.float32)
                face_area = np.array(target_landmarks[0][[20, 25, 11, 7], :]).astype(np.float32)
                xR, yR, wR, hR = target_locations[0]
                M = cv2.getPerspectiveTransform(face_area, filter_area)
                filter_warp = cv2.warpPerspective(filter_im, M, (wR, hR))

                for i in range(wR):
                    for j in range(hR):
                        modified_img[i + yR, j + xR, 0] = modified_img[i + yR, j + xR, 0] * (1-filter_warp[i, j, 3]) + filter_warp[i, j, 0] * (filter_warp[i, j, 3])
                        modified_img[i + yR, j + xR, 1] = modified_img[i + yR, j + xR, 1] * (1-filter_warp[i, j, 3]) + filter_warp[i, j, 1] * (filter_warp[i, j, 3])
                        modified_img[i + yR, j + xR, 2] = modified_img[i + yR, j + xR, 2] * (1-filter_warp[i, j, 3]) + filter_warp[i, j, 2] * (filter_warp[i, j, 3])


            oldTarget = target
            modified_img = target.copy()
            for points in current_points:
                warped_source, mask = align_source_face_to_target(source, target, points)
                if warped_source is not None:
                    modified_img = MPB(warped_source, None, mask, modified_img)
            if debug:
                plt.imshow(modified_img)
                plt.show()
            old_target = target
            old_gray = gray

            # newTarget = cv2.cvtColor(np.uint8(target * 255), cv2.COLOR_BGR2GRAY)





            #     uint_oldTarget = old_target
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
            #         #     filter_warp = (transform.warp(filterImg[:, :, :], tform3, output_shape=filterImg.shape[1::-1]))
            #         #     filter_warp[:, :, 0:2] *= 255
            #         #
            #         #     for i in range(wR):
            #         #         for j in range(hR):
            #         #             modified_img[i + yR, j + xR, 0] = modified_img[i + yR, j + xR, 0] * (1-filter_warp[i, j, 3]) + filter_warp[i, j, 0] * (filter_warp[i, j, 3])
            #         #             modified_img[i + yR, j + xR, 1] = modified_img[i + yR, j + xR, 1] * (1-filter_warp[i, j, 3]) + filter_warp[i, j, 1] * (filter_warp[i, j, 3])
            #         #             modified_img[i + yR, j + xR, 2] = modified_img[i + yR, j + xR, 2] * (1-filter_warp[i, j, 3]) + filter_warp[i, j, 2] * (filter_warp[i, j, 3])
            #         #


            '''SHOW FACE SWAPPED IMAGE'''

            # Creating video frame (this code was adapted from imageio.readthedocs.io)

            fig = plt.figure()
            plt.imshow(modified_img)

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

