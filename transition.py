# With thanks to Michiel Verbauwhede
import numpy as np
from math import log
from functools import partial, reduce

def transition_matrix(f, input_size, output_size):
    res = np.zeros((2**output_size, 2**input_size), dtype='longlong')
    for i in range(2**input_size):
        res[f(i), i] = 1
    return res

def deinterleave_rows(M, nc):
    assert(M.shape[0] == 2**(nc*int(log(M.shape[0], 2**nc))))
    for i in range(int(log(M.shape[0], 2**nc))):
        M = np.vstack([M[(2*j)*2**i:(2*j+1)*2**i, :] for j in range(M.shape[0] // 2**(i+1))] + [M[(2*j+1)*2**i:(2*j+2)*2**i, :] for j in range(M.shape[0] // 2**(i+1))])
    return M

def deinterleave(M, nc):
    M = deinterleave_rows(M, nc)
    return deinterleave_rows(M.T, nc).T

def interleave_rows(M, nc):
    assert(M.shape[0] == 2**(nc*int(log(M.shape[0], 2**nc))))
    sz = M.shape[0]//2
    for i in range(int(log(M.shape[0], 2**nc))-1, -1, -1):
        M = np.vstack(sum([[M[k*(2**i):(k+1)*2**i], M[sz+k*(2**i):sz+(k+1)*2**i]] for k in range(M.shape[0]//2**(i+1))], []))
    return M

def interleave(M, nc):
    M = interleave_rows(M, nc)
    return interleave_rows(M.T, nc).T

def FFT_like_transformation_left(M, T):
    if M.shape[0] == T.shape[1]:
        return T @ M
    assert(M.shape[0] % T.shape[1] == 0)
    ns = M.shape[0] // T.shape[1]
    MS = [FFT_like_transformation_left(M[i*ns:(i+1)*ns, :], T) for i in range(T.shape[1])]
    return np.vstack([reduce(lambda x, y: x + y, map(lambda z: z[0]*z[1], zip(T[i, :], MS))) for i in range(T.shape[0])])

def FFT_like_transformation_right(M, T):
    if M.shape[1] == T.shape[0]:
        return M @ T
    assert(M.shape[1] % T.shape[0] == 0)
    ns = M.shape[1] // T.shape[0]
    MS = [FFT_like_transformation_right(M[:, i*ns:(i+1)*ns], T) for i in range(T.shape[0])]
    return np.hstack([reduce(lambda x, y: x + y, map(lambda z: z[0]*z[1], zip(T[:, i], MS))) for i in range(T.shape[1])])

def transformed_transition_matrix(f, ni, no, T, T_inv):
    # compute T T^F T^-1
    nc = int(log(T.shape[0], 2))
    M = transition_matrix(f, ni, no)
    M_big = M
    for _ in range(1, nc):
        M_big = np.kron(M_big, M)
    if nc > 1:
        M_big = interleave(M_big, nc)
    # do the FFT-like operation
    M_big = FFT_like_transformation_left(M_big, T)
    M_big = FFT_like_transformation_right(M_big, T_inv)
    if nc > 1:
        M_big = deinterleave(M_big, nc)
    return M_big

T_linear = np.array([[1, 1],[1, -1]], dtype=np.float64)
correlation_matrix = partial(transformed_transition_matrix, T=T_linear, T_inv = np.linalg.inv(T_linear))
T_quasi_differential = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, -1, 1, 0]], dtype=np.float64)
quasi_differential_transition_matrix = partial(transformed_transition_matrix, T=T_quasi_differential, T_inv = np.linalg.inv(T_quasi_differential))
T_ultrametric = np.array([[1, 1],[0, 1]], dtype=np.float64)
ultrametric_transition_matrix = partial(transformed_transition_matrix, T=T_ultrametric, T_inv = np.linalg.inv(T_ultrametric))
