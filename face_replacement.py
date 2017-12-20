import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform as transform
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg
from modified_poisson_blending import modified_poisson_blending as MPB
from face_landmark import align_source_face_to_target, get_face_landmarks, detect_face

#https://matthewearl.github.io/2015/07/28/switching-eds-with-python/

def to_gray(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def face_replacement(source_vid, target_vid, out_filename, filter_im_orig, debug=False):

    source = source_vid.get_next_data()
    target = target_vid.get_next_data()

    s_fps = source_vid._meta['fps']
    t_fps = target_vid._meta['fps']

    trackedVideo = imageio.get_writer(out_filename, fps=s_fps)

    lk_params = dict(winSize=(100, 100),
                     maxLevel=15,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_gray = to_gray(target)

    points_list = []
    for i, (source, target) in enumerate(zip(source_vid, target_vid)):

        if i == 79:
            print "bug"


        #source_landmarks, source_locations = get_face_landmarks(source)
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

        old_gray = gray

        can_swap = (detect_face(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)) != 0 and len(target_landmarks) != 0)

        if not can_swap:
            modified_img = target
        else:

            modified_img = target.copy()
            for points in current_points:
                warped_source, mask = align_source_face_to_target(source, target, points)
                if warped_source is not None:
                    modified_img = MPB(warped_source, mask, modified_img)
            if debug:
                plt.imshow(modified_img)
                plt.show()
            if filter_im_orig is not None:
                filter_im = np.array(filter_im_orig*255).astype(np.uint8)
                face_area = np.array(target_landmarks[0][[19, 25, 11, 6], :]).astype(np.float32)

                face_area[:, 0] -= min(face_area[:, 0])
                face_area[:, 1] -= min(face_area[:, 1])
                h = (np.max(face_area[:,0])).astype(np.int)
                h += np.round(h * 0.3).astype(np.int)
                w = (np.max(face_area[:,1])).astype(np.int)
                filter_im = resize(filter_im, (h + np.round(h * 0.3), w, filter_im.shape[2]))


                filter_area = np.array([[0,0],[h, 0],[h, w], [0,w]]).astype(np.float32)

                M = cv2.getPerspectiveTransform(filter_area, face_area)
                # h += np.round(h * 0.3).astype(np.int)
                filter_warp_uncropped = cv2.warpPerspective(filter_im, M, (w, h))

                r_crop, c_crop = np.where(filter_warp_uncropped[:, :, 0] != 0)
                filter_warp = filter_warp_uncropped[np.min(r_crop) : np.max(r_crop), np.min(c_crop) : np.max(c_crop)]


                filter_warp = resize(filter_warp, (np.round(filter_warp.shape[0]*1.5), np.round(filter_warp.shape[1]*2), filter_warp.shape[2]))
                xR, yR, wR, hR = target_locations[0]
                yR -= 60

                yR = max(0, yR)

                r = filter_warp.shape[0]
                c = filter_warp.shape[1]

                if (r + yR) >= modified_img.shape[0]:
                    r = (modified_img.shape[0] - yR)
                if (c + xR) >= modified_img.shape[1]:
                    c = (modified_img.shape[1] - xR)

                modified_img[yR: r + yR, xR: c + xR, 0] = modified_img[yR: r + yR, xR: c + xR, 0] * (1 - filter_warp[0:r, 0:c, 3]) + filter_warp[0:r, 0:c, 0] * 255 * (filter_warp[0:r, 0:c, 3])
                modified_img[yR: r + yR, xR: c + xR, 1] = modified_img[yR: r + yR, xR: c + xR, 1] * (1 - filter_warp[0:r, 0:c, 3]) + filter_warp[0:r, 0:c, 1] * 255 * (filter_warp[0:r, 0:c, 3])
                modified_img[yR: r + yR, xR: c + xR, 2] = modified_img[yR: r + yR, xR: c + xR, 2] * (1 - filter_warp[0:r, 0:c, 3]) + filter_warp[0:r, 0:c, 2] * 255 * (filter_warp[0:r, 0:c, 3])


            oldTarget = target
            for points in current_points:
                warped_source, mask = align_source_face_to_target(source, target, points)
                if warped_source is not None:
                    modified_img = MPB(warped_source, mask, modified_img)
            if debug:
                plt.imshow(modified_img)
                plt.show()
            old_gray = gray


        # Creating video frame (this code was adapted from imageio.readthedocs.io)

        fig = plt.figure()
        plt.imshow(modified_img)
        # plt.show()

        canvas = plt.get_current_fig_manager().canvas
        agg = canvas.switch_backends(FigureCanvasAgg)
        agg.draw()
        s = agg.tostring_rgb()
        l, b, w, h = agg.figure.bbox.bounds
        w, h = int(w), int(h)
        buf = np.fromstring(s, dtype=np.uint8)
        buf.shape = h, w, 3
        trackedVideo.append_data(buf)
        plt.close()

        print "Frame", i


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

