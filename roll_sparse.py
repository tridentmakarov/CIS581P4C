import scipy.sparse
import numpy as np
#Function copied from
# https://github.com/librosa/librosa/blob/ad5ccffcd067ea2048e096533daf93d0d247afa6/librosa/util/utils.py

def roll_sparse(x, shift, axis=0):
    if not scipy.sparse.isspmatrix(x):
        return np.roll(x, shift, axis=axis)

    # shift-mod-length lets us have shift > x.shape[axis]
    if axis not in [0, 1, -1]:
        raise ValueError('axis must be one of (0, 1, -1)')

    shift = np.mod(shift, x.shape[axis])

    if shift == 0:
        return x.copy()

    fmt = x.format
    if axis == 0:
        x = x.tocsc()
    elif axis in (-1, 1):
        x = x.tocsr()

    # lil matrix to start
    x_r = scipy.sparse.lil_matrix(x.shape, dtype=x.dtype)

    idx_in = [slice(None)] * x.ndim
    idx_out = [slice(None)] * x_r.ndim

    idx_in[axis] = slice(0, -shift)
    idx_out[axis] = slice(shift, None)
    x_r[tuple(idx_out)] = x[tuple(idx_in)]

    idx_out[axis] = slice(0, shift)
    idx_in[axis] = slice(-shift, None)
    x_r[tuple(idx_out)] = x[tuple(idx_in)]

    return x_r.asformat(fmt)
