import math

import numpy as np
from numba import cuda

TPB = 16


@cuda.reduce
def calc_error(a, b):
    return a + b


@cuda.jit(device=True)
def move(ins1, ins2, distance_matrix, projection, learning_rate, _error):
    size = len(projection)
    total = len(distance_matrix)
    n_dimensions = int(projection[ins1].shape[0])

    x1x2 = projection[ins2][0] - projection[ins1][0]
    y1y2 = projection[ins2][1] - projection[ins1][1]
    if n_dimensions == 3:
        z1z2 = projection[ins2][2] - projection[ins1][2]
    dr2 = max(math.sqrt(x1x2 * x1x2 + y1y2 * y1y2), 0.0001)

    # getting te index in the distance matrix and getting the value
    r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(i,j)
    s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(i,j)
    drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

    # calculate the movement
    delta = (drn - dr2)
    _error = math.fabs(delta)

    # moving
    projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
    projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)
    if n_dimensions == 3:
        projection[ins2][1] += learning_rate * delta * (z1z2 / dr2)

    return _error / size


@cuda.jit
def iteration(index, distance_matrix, projection, learning_rate, g_error):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    # Block id in a 1D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    ins1 = index[tx + bx * bw]
    ins2 = index[ty + by * bh]
    if tx + bx * bw == ty + by * bh:
        return

    error = int(0)
    error = move(ins1, ins2, distance_matrix, projection, learning_rate, error)
    g_error[tx + bx * bw, ty + by * bh] = error


def gpu_transform(nfs, X, *, index, total, inplace):
    # iterate until max_it or if the error does not change more than the tolerance
    error = math.inf

    d_distance_matrix = cuda.to_device(nfs.embedding_)
    d_index = cuda.to_device(index)
    d_projection = cuda.to_device(X)

    size = int(math.sqrt(2 * total + 1))
    threadsperblock = nfs.cuda_threads_per_block if nfs.cuda_threads_per_block else (TPB, TPB)
    if nfs.cuda_blocks_per_grid is not None:
        blockspergrid = nfs.cuda_blocks_per_grid
    elif isinstance(threadsperblock, int):
        blockspergrid = (size) // threadsperblock
    else:
        blockspergrid = ((size) // threadsperblock[0], (size) // threadsperblock[1])
    nfs.print(f'Dist: threads:{threadsperblock}, blocks:{blockspergrid}')

    if nfs.cuda_profile:
        cuda.profile_start()

    ref_new_error = np.zeros(shape=(size, size), dtype=np.float64)
    d_new_error = cuda.to_device(ref_new_error)
    for k in range(nfs.max_it):
        learning_rate = nfs.learning_rate0 * math.pow((1 - k / nfs.max_it), nfs.decay)
        iteration[blockspergrid, threadsperblock](d_index, d_distance_matrix, d_projection, learning_rate,
                                                  d_new_error)
        new_error = calc_error(d_new_error.reshape(size * size)) / (size)
        # d_new_error.copy_to_host(ref_new_error)
        # new_error = ref_new_error.sum()/size
        # self.print(new_error, error, size, np.isinf(ref_new_error).sum(), np.isnan(ref_new_error).sum())
        if math.fabs(new_error - error) < nfs.tolerance:
            nfs.print(f'Error below tolerance {math.fabs(new_error - error)} in iteration {k}, breaking')
            break

        error = new_error
    if nfs.cuda_profile:
        cuda.profile_stop()
    nfs.print(f'Iter: {k + 1} error: {error}')

    if inplace:
        d_projection.copy_to_host(X)
        return X, error
    else:
        tmp = None
        d_projection.copy_to_host(tmp)
        return tmp, error
