import math
import pickle
from typing import Optional, Tuple
from scipy.stats import norm

import numba

import numpy as np
from sklearn.preprocessing import MinMaxScaler

MACHINE_EPSILON = np.finfo(np.double).eps


def read_distance_matrix(filename: str) -> [float, np.array]:
    # calculating the size
    with open(filename) as fp:
        line = fp.readline()
    tokens = line.strip().split(' ')

    size = len(tokens)
    distance_matrix = np.zeros(int(size * (size + 1) / 2), dtype=np.float32)

    # reading line per line
    row = 0
    k = 0
    with open(filename) as fp:
        line = fp.readline()
        while line:
            tokens = line.strip().split(' ')
            for column in range(row, size):
                distance_matrix[k] = float(tokens[column])
                k = k + 1
            line = fp.readline()
            row = row + 1
    return size, distance_matrix


def pickle_load_matrix(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]


def pickle_save_matrix(filename, distance_matrix, size):
    with open(filename, 'wb') as f:
        pickle.dump([distance_matrix, size], f)

@numba.njit(parallel=True, fastmath=True)
def move(ins1, distance_matrix, projection, learning_rate, n_dimension, metric,
         force_projection_dimensions=None,
         original_z_axis=None,
         z_axis_moving_range: Optional[Tuple[float, float]] = (0, 0),
         range_strict_limitation: Optional[bool] = True,
         z_score: Optional[float] = 1):
    size = len(projection)
    total = len(distance_matrix)
    error = 0

    for ins2 in numba.prange(size):
        if ins1 != ins2:
            temp_dist = np.zeros(n_dimension)
            temp_dr2 = 0
            for index in range(n_dimension):
                temp_dist[index] = projection[ins2][index] - projection[ins1][index]
                temp_dr2 += temp_dist[index] * temp_dist[index]
            dr2 = max(math.sqrt(temp_dr2), 0.0001)
            # dr2 = max(metric(projection[ins1], projection[ins2]), 0.0001)

            # getting te index in the distance matrix and getting the value
            r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(i,j,k)
            s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(i,j,k)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            # calculate the movement
            delta = (drn - dr2)
            error += math.fabs(delta)

            inst_z = z_score
            theta = (z_axis_moving_range[0] / inst_z)
            possibility_to_ratio = (theta * math.sqrt(2 * math.pi))

            # If fixing z axis, only move x and y
            if force_projection_dimensions is not None:
                for index in force_projection_dimensions:
                    projection[ins2][index] += learning_rate * delta * (temp_dist[index] / dr2)

                    z_axis_difference = projection[ins2][-1] + learning_rate * delta * (temp_dist[index] / dr2) - \
                                        original_z_axis[ins2]

                    if not z_axis_moving_range == (0, 0):
                        if range_strict_limitation:
                            if z_axis_moving_range[0] <= z_axis_difference <= z_axis_moving_range[1]:
                                projection[ins2][-1] += learning_rate * delta * (temp_dist[index] / dr2)

                        else:
                            moving_to_ratio = (1 / (math.sqrt(2 * math.pi) * theta)) \
                                              * math.exp(-((z_axis_difference) ** 2
                                                           / (2 * (theta ** 2)))) * possibility_to_ratio
                            projection[ins2][-1] += learning_rate * delta * (temp_dist[index] / dr2) * moving_to_ratio

            else:
                for index in range(n_dimension):
                    projection[ins2][index] += learning_rate * delta * (temp_dist[index] / dr2)

    return error / size


@numba.njit(parallel=True, fastmath=True)
def iteration(index, distance_matrix, projection, learning_rate, n_dimension, metric,
              force_projection_dimensions=None, original_z_axis=None,
              z_axis_moving_range: Optional[Tuple[float, float]] = (0, 0),
              z_score: Optional[float] = 1,
              range_strict_limitation: Optional[bool] = True):
    size = len(projection)
    error = 0

    for i in numba.prange(size):
        ins1 = index[i]
        error += move(ins1=ins1,
                      distance_matrix=distance_matrix,
                      projection=projection,
                      learning_rate=learning_rate,
                      n_dimension=n_dimension,
                      metric=metric,
                      force_projection_dimensions=force_projection_dimensions,
                      original_z_axis=original_z_axis,
                      z_axis_moving_range=z_axis_moving_range,
                      z_score=z_score,
                      range_strict_limitation=range_strict_limitation)

    return error / size


@numba.njit(parallel=True, fastmath=True)
def create_triangular_distance_matrix(
        data,
        metric):
    distance_matrix = np.zeros(int((data.shape[0] + 1) * data.shape[0] / 2))
    size = len(data)

    k = 0
    for i in range(size):
        for j in range(i, size):
            distance_matrix[k] = metric(data[i], data[j])
            k = k + 1

    return distance_matrix


@numba.njit(parallel=True)
def kruskal_stress(distance_matrix, projection, metric):
    size = len(projection)
    total = len(distance_matrix)

    den = 0
    num = 0

    for i in numba.prange(size):
        for j in numba.prange(size):
            dr2 = metric(projection[i], projection[j])

            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            num += (drn - dr2) * (drn - dr2)
            den += drn * drn

    return math.sqrt(num / den)


def scale_dataset(data, feature_range):
    """
    Helper function for scaling/normalizing
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    data = scaler.fit_transform(data)
    """
    data_scalared = data
    scaler = StandardScaler()
    data = scaler.fit_transform(data, data_scalared)
    """
    return data


def non_numeric_processor(data, axes=None):
    """
    Helper function for processing numeric columns
    """
    for col in axes:
        non_numeric = [data[0][col]]
        data[0][col] = 0

        # if the name(string) appears the first time, add it into the name list
        # then assign it an integer
        # if it has shown before, change it into its assigned integer

        for index in range(1, len(data)):
            if data[index][col] not in non_numeric:
                non_numeric.append(data[index][col])
            data[index][col] = non_numeric.index(data[index][col])
    return data
