import math
from numba import njit, prange


@njit(parallel=True)
def kruskal_stress(distance_matrix, projection):
    size = len(projection)
    total = len(distance_matrix)

    den = 0
    num = 0

    for i in prange(size):
        for j in prange(size):
            dr2 = math.sqrt((projection[i][0] - projection[j][0]) * (projection[i][0] - projection[j][0]) +
                            (projection[i][1] - projection[j][1]) * (projection[i][1] - projection[j][1]))

            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            num += (drn - dr2) * (drn - dr2)
            den += drn * drn

    return math.sqrt(num / den)
