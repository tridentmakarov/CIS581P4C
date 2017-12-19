import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import copy
import skimage.transform
import imutils
from imutils import face_utils

predictor_path = "resources/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def get_face_landmarks(image, debug=False):

    if debug==True:
        plt.imshow(image)
        plt.show()
    
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    im_shape = []
    locations = []
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        im_shape.append(face_utils.shape_to_np(predictor(gray, rect)))

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        locations.append(face_utils.rect_to_bb(rect))
    return im_shape, locations


#Based partly on http://www.learnopencv.com/face-morph-using-opencv-cpp-python/ and other articles on website
def align_source_face_to_target(source_im, target_im, tracked_points=None, opt_flow_usage_factor=0.66, debug=False):    # https: // www.pyimagesearch.com / 2017 / 04 / 03 / facial - landmarks - dlib - opencv - python /
    source_landmarks, source_locations = get_face_landmarks(source_im)
    target_landmarks, target_locations = get_face_landmarks(target_im)

    source_landmarks = source_landmarks[0]
    target_landmarks = target_landmarks[0]

    if tracked_points is not None:
        pts = tracked_points["points"]
        state = tracked_points['good_points']
        target_landmarks[state] = (pts[state] * opt_flow_usage_factor) + \
                      (target_landmarks[state] * (1 - opt_flow_usage_factor))

    if debug:
        plt.imshow(source_im)
        plt.scatter(source_landmarks[:,0], source_landmarks[:,1])
        plt.show()
        plt.imshow(target_im)
        plt.scatter(target_landmarks[:,0], target_landmarks[:,1])
        plt.show()

    source_convex_hull = cv2.convexHull(source_landmarks, returnPoints = False)
    target_convex_hull = source_convex_hull
    # source_hull_points = np.squeeze(source_landmarks[source_convex_hull])
    # target_hull_points = np.squeeze(target_landmarks[target_convex_hull])

    transform = skimage.transform.PiecewiseAffineTransform()
    transform.estimate(target_landmarks, source_landmarks)
    #transform.estimate(target_hull_points, source_hull_points)

    source_mask = np.concatenate((source_im, np.full(source_im.shape[:2] + (1,), 255, dtype=np.uint8)), axis=2)
    warped_source_mask = skimage.transform.warp(source_mask, transform, output_shape=target_im.shape[:2])

    warp = warped_source_mask[:,:,:3]
    mask = warped_source_mask[:,:,3]
    if debug:
        plt.imshow(warp)
        plt.show()
        plt.imshow(mask)
        plt.show()
    return warp, mask
