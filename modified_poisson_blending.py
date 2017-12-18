import cv2
import numpy as np
import matplotlib.pyplot as plt

def modified_poisson_blending(source_face, target_face, mask, original_target, location):
    plt.imshow(source_face)
    plt.show()
    #(x, y, w, h) = location
    (x, y) = (location[0], location[1])
    mask = mask.astype(np.uint8) * 255
    print source_face.dtype
    if not source_face.dtype == np.uint8:
        print "Changing dtype"
        source_face = source_face.copy()
        source_face *= 255
        source_face = source_face.astype(np.uint8)

    #center = (x + w//2 + 10, y + h//2 + 10)
    center = (original_target.shape[0]//2, original_target.shape[1]//2)
    #return cv2.seamlessClone(source_face, original_target, mask, center, cv2.MIXED_CLONE)
    return cv2.seamlessClone(source_face, original_target, mask, center, cv2.MIXED_CLONE)
