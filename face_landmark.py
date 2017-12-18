import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import copy

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

    source_convex_hull = cv2.convexHull(source_landmarks, returnPoints = False)
    target_convex_hull = cv2.convexHull(target_landmarks, returnPoints = False)
    source_hull_points = np.squeeze(source_landmarks[source_convex_hull])
    target_hull_points = np.squeeze(target_landmarks[target_convex_hull])

    source_hull_points[:, 0] = np.clip(source_hull_points[:, 0], 0, source_im.shape[0] - 1)
    source_hull_points[:, 1] = np.clip(source_hull_points[:, 1], 0, source_im.shape[1] - 1)
    target_hull_points[:, 0] = np.clip(target_hull_points[:, 0], 0, target_im.shape[0] - 1)
    target_hull_points[:, 1] = np.clip(target_hull_points[:, 1], 0, target_im.shape[1] - 1)

    plt.imshow(source_im)
    plt.scatter(source_hull_points[:, 0], source_hull_points[:, 1])
    plt.show()
    source_delaunay = Delaunay(source_hull_points)
    target_delaunay = copy.deepcopy(source_delaunay)
    target_delaunay.points = target_hull_points.astype(np.float)
    transforms = [cv2.getAffineTransform(source_hull_points[simplex].astype(np.float32),
                                         target_hull_points[simplex].astype(np.float32))
                  for simplex in source_delaunay.simplices]

    warped_source = np.full(source_im.shape, 255, source_im.dtype)

    #Based on https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
    for simplex, transform in zip(source_delaunay.simplices, transforms):
        source_tri = source_hull_points[simplex]
        target_tri = target_hull_points[simplex]
        source_bound = list(cv2.boundingRect(source_tri))
        target_bound = list(cv2.boundingRect(target_tri))

        source_tri_cropped = []
        target_tri_cropped = []
        for i in xrange(0, 3):
            source_tri_cropped.append(((source_tri[i, 0] - source_bound[0]), (source_tri[i, 1] - source_bound[1])))
            target_tri_cropped.append(((target_tri[i, 0] - target_bound[0]), (target_tri[i, 1] - target_bound[1])))

        im_triangle = source_im[source_bound[1]:source_bound[1] + source_bound[3], source_bound[0]:source_bound[0] + source_bound[2]]

        im_triangle = cv2.warpAffine(im_triangle, transform, (target_bound[2], target_bound[3]), None, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((target_bound[3], target_bound[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(target_tri_cropped), (1.0, 1.0, 1.0), 16, 0)

        im_triangle *= mask

        warped_source[target_bound[1]:target_bound[1] + target_bound[3], target_bound[0]:target_bound[0] + target_bound[2]] *=\
            (1.0, 1.0, 1.0) - mask

        warped_source[target_bound[1]:target_bound[1] + target_bound[3], target_bound[0]:target_bound[0] + target_bound[2]] +=\
            im_triangle

        plt.imshow(warped_source)
        plt.show()

    return warped_source

