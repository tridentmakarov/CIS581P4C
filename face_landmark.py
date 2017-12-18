import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import copy
import skimage.transform

predictor_path = "resources/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def get_face_landmarks(img, rect):
    (x,y, w, h) = rect
    dlib_rect = dlib.rectangle(x,y, x + w, y + h)
    print img.shape
    print img.dtype
    shape = predictor(img.copy(), dlib_rect) #Using img.copy b/c of bug https://github.com/davisking/dlib/issues/128
    return [(p.x, p.y) for p in shape.parts()]


#Based partly on http://www.learnopencv.com/face-morph-using-opencv-cpp-python/ and other articles on website
def align_source_face_to_target(source_im, target_im):
    source_landmarks = np.array(get_face_landmarks(source_im, (0,0, source_im.shape[1], source_im.shape[0])))
    target_landmarks = np.array(get_face_landmarks(target_im, (0,0, target_im.shape[1], target_im.shape[0])))

    plt.imshow(source_im)
    plt.scatter(source_landmarks[:,0], source_landmarks[:,1])
    plt.show()
    plt.imshow(target_im)
    plt.scatter(target_landmarks[:,0], target_landmarks[:,1])
    plt.show()

    source_convex_hull = cv2.convexHull(source_landmarks, returnPoints = False)
    #target_convex_hull = cv2.convexHull(target_landmarks, returnPoints = False)
    target_convex_hull = source_convex_hull
    source_hull_points = np.squeeze(source_landmarks[source_convex_hull])
    target_hull_points = np.squeeze(target_landmarks[target_convex_hull])

    source_hull_points[:, 0] = np.clip(source_hull_points[:, 0], 0, source_im.shape[0] - 1)
    source_hull_points[:, 1] = np.clip(source_hull_points[:, 1], 0, source_im.shape[1] - 1)
    target_hull_points[:, 0] = np.clip(target_hull_points[:, 0], 0, target_im.shape[0] - 1)
    target_hull_points[:, 1] = np.clip(target_hull_points[:, 1], 0, target_im.shape[1] - 1)

    transform = skimage.transform.PiecewiseAffineTransform()
    transform.estimate(source_hull_points, target_hull_points)
    warp = skimage.transform.warp(source_im, transform)

    return warp


