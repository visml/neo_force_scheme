import math

from numba import cuda


@cuda.reduce
def calc_error(a, b):
    return a + b


@cuda.jit(device=True)
def move(ins1, ins2, distance_matrix, projection, learning_rate, _error):
    size = len(projection)
    total = len(distance_matrix)

    x1x2 = projection[ins2][0] - projection[ins1][0]
    y1y2 = projection[ins2][1] - projection[ins1][1]
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

    error = 0
    error = move(ins1, ins2, distance_matrix, projection, learning_rate, error)
    g_error[tx + bx * bw, ty + by * bh] = error
