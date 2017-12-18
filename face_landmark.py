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

def get_face_landmarks(image, rect):
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
    
    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(args["image"])
    
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        im_shape = predictor(gray, rect)
        im_shape = face_utils.shape_to_np(im_shape)
        
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # show the face number
        
        
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
    return im_shape


#Based partly on http://www.learnopencv.com/face-morph-using-opencv-cpp-python/ and other articles on website
def align_source_face_to_target(source_im, target_im):
    # https: // www.pyimagesearch.com / 2017 / 04 / 03 / facial - landmarks - dlib - opencv - python /
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

    # source_hull_points[:, 0] = np.clip(source_hull_points[:, 0], 0, source_im.shape[0] - 1)
    # source_hull_points[:, 1] = np.clip(source_hull_points[:, 1], 0, source_im.shape[1] - 1)
    # target_hull_points[:, 0] = np.clip(target_hull_points[:, 0], 0, target_im.shape[0] - 1)
    # target_hull_points[:, 1] = np.clip(target_hull_points[:, 1], 0, target_im.shape[1] - 1)

    transform = skimage.transform.PiecewiseAffineTransform()
    transform.estimate(source_hull_points, target_hull_points)
    warp = skimage.transform.warp(source_im, transform)
    

    return warp


