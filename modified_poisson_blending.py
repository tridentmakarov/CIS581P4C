import cv2
import numpy as np
import scipy.misc as sc
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from skimage.transform import resize

def modified_poisson_blending(source, target, mask, originalTarget, x_corner, y_corner):
    source /= 255
    target /= 255
    mask /= 255

    F = np.zeros(source.shape)

    F = 1 - F

    return output_img


# Poisson blending using http://vacation.aid.design.kyushu-u.ac.jp/and/poisson/
def poisson_gray(source, target, mask):
    n = source.size
    A = sparse(n, n)
    f = np.zeros(n, 1)
    fx = np.where(mask > 0)
    bx = np.where(mask == 0)
    q = np.zeros(n, 1)
    q[fx] = 1

    I = np.diag(sparse(q))
    A = -4 * I

    return F