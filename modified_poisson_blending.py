import cv2
import numpy as np
import scipy.misc
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from skimage.transform import resize

def modified_poisson_blending(source_face, target, mask, originalTarget, x_corner, y_corner):
    source_face /= 255.0
    target /= 255.0
    mask /= 255.0

    F = np.zeros(source_face.shape)
    for i in range(3):
        F[:,:,i] = poisson_gray(source_face[:,:,i], target[:,:,i], mask)
    np.logical_not(mask, out=mask)

    for i in range(3):
        F[:,:,i] = poisson_gray(target[:,:,i], F[:,:,i], mask)

    modified_img = target.copy()
    modified_img[x_corner:x_corner + source_face.shape[0],
                y_corner:y_corner + source_face.shape[1],:] = (F * 255).astype(np.uint8)

    return modified_img


# Poisson blending using http://vacation.aid.design.kyushu-u.ac.jp/and/poisson/
def poisson_gray(source, target, mask):
    n = source.size
    f = np.zeros(n, 1)
    fx = mask > 0
    bx = mask == 0
    q = np.zeros(n, 1)
    q[fx] = 1

    I = np.diag(sparse.csc_matrix(q))
    A = -4 * I
    A += np.roll(I, (0, source.shape[0]), (0,1)) + np.roll(I, (0, - source.shape[0]), (0,1)) +\
         np.roll(I, (0, 1), (0,1)) + np.roll(I, (0, -1), (0,1))
    A += sparse.eye(n) - I
    b = np.zeros(n, 1)
    b[bx] = target[bx]

    laplacian_target = np.roll(target, (1, 0), (0,1)) + np.roll(target, (1, 0), (0,-1)) +\
                       np.roll(target, (0, 1), (0,1)) + np.roll(target, (0, 1), (0,-1))
    laplacian_target -= 4*(target.astype(np.int))
    b[fx] = laplacian_target[fx]
    x = np.linalg.solve(A, b)
    return x.reshape(source.shape)
