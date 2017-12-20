import cv2
import numpy as np
import matplotlib.pyplot as plt

def modified_poisson_blending(source_face, mask, original_target):
    mask = mask.astype(np.uint8) * 255
    if not source_face.dtype == np.uint8:
        source_face = source_face.copy()
        source_face *= 255
        source_face = source_face.astype(np.uint8)

    unmasked_pixels_y, unmasked_pixels_x = np.where(mask > 0)

    center = (np.mean([np.max(unmasked_pixels_x), np.min(unmasked_pixels_x)]).astype(int),
              np.mean([np.max(unmasked_pixels_y), np.min(unmasked_pixels_y)]).astype(int))
    return cv2.seamlessClone(source_face, original_target, mask, center, cv2.NORMAL_CLONE)
