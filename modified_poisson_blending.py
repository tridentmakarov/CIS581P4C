import cv2
import numpy as np
import scipy.misc
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from skimage.transform import resize
from roll_sparse import roll_sparse

def modified_poisson_blending(source_face, target, mask, originalTarget, x_corner, y_corner):
    # if not source_face.dtype == float:
    #     source_face = source_face.astype(float)
    #     source_face /= 255
    # if not target.dtype == float:
    #     target = target.astype(float)
    #     target /= 255
    #mask = mask.astype(float)
    mask = mask.astype(np.uint8) * 255
    print source_face.dtype
    if not source_face.dtype == np.uint8:
        print "Changing dtype"
        source_face *= 255
        source_face = source_face.astype(np.uint8)

    if not target.dtype == np.uint8:
        target *= 255
        target = target.astype(np.uint8)

    #center = (x_corner + target.shape[0]//2, y_corner + target.shape[1]//2)
    center = (target.shape[0]//2, target.shape[1]//2)
    #modified_img = cv2.seamlessClone(source_face, originalTarget, mask, center,cv2.MIXED_CLONE)
    out = cv2.seamlessClone(source_face, target, mask, center,cv2.MIXED_CLONE)
    out = cv2.seamlessClone(target, out, ~mask, center,cv2.NORMAL_CLONE)
    modified_img = originalTarget.copy()
    modified_img[y_corner:y_corner + target.shape[1], x_corner:x_corner + target.shape[0]] = out
    #blended_target = originalTarget[x_corner:target.shape[0], y_corner:target.shape[1], :]
    #blended_target = originalTarget[x_corner:x_corner+target.shape[0], y_corner:y_corner+target.shape[1], :]
    #blended_target = originalTarget[y_corner:y_corner+target.shape[1], x_corner:x_corner+target.shape[0], :]
    # modified_img = cv2.seamlessClone(blended_target,
    #                                  modified_img, mask, center,cv2.NORMAL_CLONE)

    return modified_img.astype(np.uint8)

    F = np.zeros(source_face.shape)
    for i in range(3):
        F[:,:,i] = poisson_gray(source_face[:,:,i], target[:,:,i], mask)
    np.logical_not(mask, out=mask)

    for i in range(3):
        F[:,:,i] = poisson_gray(target[:,:,i], F[:,:,i], mask)
    plt.imshow(F/(-1e16))
    plt.show()
    modified_img = originalTarget.copy()
    modified_img[y_corner:y_corner + source_face.shape[1],
                x_corner:x_corner + source_face.shape[0],:] = (F * 255).astype(np.uint8)

    return modified_img


# Poisson blending using http://vacation.aid.design.kyushu-u.ac.jp/and/poisson/
def poisson_gray(source, target, mask):
    n = source.size
    #f = np.zeros((n, 1))
    fx = mask != 0
    bx = mask == 0
    q = np.zeros(n)
    q[fx.flatten()] = 1
    #I = scipy.sparse.diags(q)
    I = np.diag(q)
    A = -4 * I
    print "Sparse roll"
    # A += roll_sparse(I, source.shape[0], 1) + \
    #      roll_sparse(I, -source.shape[0], 1) + \
    #      roll_sparse(I, 1, 1) + roll_sparse(I, -1, 1)
    # A += sparse.eye(n) - I
    A += np.roll(I, (0, source.shape[0]), (0,1)) + np.roll(I, (0, - source.shape[0]), (0,1)) +\
         np.roll(I, (0, 1), (0,1)) + np.roll(I, (0, -1), (0,1))
    A += np.eye(n) - I
    b = np.zeros(n)
    b[bx.flatten()] = target[bx]

    print "Roll"
    laplacian_target = np.roll(target, (1, 0), (0,1)) + np.roll(target, (1, 0), (0,-1)) +\
                       np.roll(target, (0, 1), (0,1)) + np.roll(target, (0, 1), (0,-1))
    laplacian_target -= 4*(target.astype(np.int))
    b[fx.flatten()] = laplacian_target[fx]
    x = sparse.linalg.spsolve(A, b)
    return x.reshape(source.shape)
