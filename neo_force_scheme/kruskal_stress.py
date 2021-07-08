import math
from typing import Optional

from numba import njit, prange


@njit(parallel=True)
def kruskal_stress(distance_matrix, projection, n_dimension: Optional[int] = 2):
    size = len(projection)
    total = len(distance_matrix)

    den = 0
    num = 0

    for i in prange(size):
        for j in prange(size):
            temp_dr2 = 0
            for index in range(n_dimension):
                temp_dr2 += (projection[i][index] - projection[j][index]) * (projection[i][index] - projection[j][index])
            dr2 = math.sqrt(temp_dr2)

            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            num += (drn - dr2) * (drn - dr2)
            den += drn * drn

    return math.sqrt(num / den)
