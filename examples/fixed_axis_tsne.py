# coding:utf-8

import math
from datetime import timedelta
from timeit import default_timer as timer
from typing import Optional, List, Tuple

import numba
import numpy as np
import sklearn.datasets as datasets
from sklearn.preprocessing import MinMaxScaler


def tsne(x, no_dims=2, perplexity=30.0, max_iter=1000,
         fix_column_to_z_projection_axis: Optional[int] = None,
         drop_columns_from_dataset: Optional[List[int]] = None,
         scaler: Optional[Tuple[int, int]] = (0, 1)):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """
    start = timer()

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1

    if scaler is not None:
        x = scale_dataset(x, scaler)

    z_axis_fixed = x[:, fix_column_to_z_projection_axis]
    x = preprocess_data(data=x, drop_columns_from_dataset=drop_columns_from_dataset)

    (n, d) = x.shape

    # Momentum
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # Randomly initiate y
    y = np.random.randn(n, no_dims)

    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))

    gains = np.ones((n, no_dims))

    # Symmetrize
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)  # pij
    # early exaggeration

    P = P * 4
    P = np.maximum(P, 1e-12)

    if fix_column_to_z_projection_axis is not None:
        force_projection_dimensions = np.arange(no_dims - 1)
        y[:, no_dims - 1] = z_axis_fixed
    else:
        force_projection_dimensions = np.arange(no_dims)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)  # qij
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        # pij-qij
        PQ = P - Q

        for i in range(n):
            dy[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        # iteration
        iy = momentum * iy - eta * (gains * dy)
        for inst in range(len(y)):
            for index in force_projection_dimensions:
                y[inst][index] = y[inst][index] + iy[inst][index]
                y[inst][index] = y[inst][index] - np.tile(np.mean(y, 0), (n, 1))[inst][index]
        # # Compute current value of cost function\
        # if (iter + 1) % 100 == 0:
        #     C = np.sum(P * np.log(P / Q))
        #     print("Iteration ", (iter + 1), ": error is ", C)
        #     if (iter+1) != 100:
        #         ratio = C/oldC
        #         print("ratio ", ratio)
        #         if ratio >= 0.95:
        #             break
        #     oldC = C
        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    end = timer()
    print(f'Time elapsed: {timedelta(seconds=end - start)}')

    return y


# Below are the functions used during tsne processing

def cal_pairwise_dist(x):
    '''Calculate the distance of a pairwise, x is a matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist


def cal_perplexity(dist, idx=0, beta=1.0):
    '''Calculate perplexity. D is distance vector
    idx is the distance between a point and itself,
    beta is Gaussian distribution parameter
    '''
    prob = np.exp(-dist * beta)

    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob < 1e-12:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
        prob /= sum_prob

    return perp, prob


def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''Using binary research to find beta,
    then calculate the prob of the pairwise
    '''

    # initialize parameters
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    dist[dist < 0] = 0
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # Here use the log value to make the later calculation easier
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." % (i, n))

        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # Using binary research to find the prob under the best sigma
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # update the value for perb and prob
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # record the value for prob
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))

    return pair_prob


def create_triangular_distance_matrix(
        data):
    distance_matrix = np.zeros(int((data.shape[0] + 1) * data.shape[0] / 2))
    size = len(data)

    k = 0
    for i in range(size):
        for j in range(i, size):
            distance_matrix[k] = euclidean(data[i], data[j])
            k = k + 1

    return distance_matrix


def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


def kruskal_stress(distance_matrix, projection):
    size = len(projection)
    total = len(distance_matrix)

    den = 0
    num = 0

    for i in numba.prange(size):
        for j in numba.prange(size):
            dr2 = euclidean(projection[i], projection[j])

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


def preprocess_data(data,
                    drop_columns_from_dataset=None):
    """
    Helper function for preprocessing the data
    :param data:
    :param fixed_axis:
    :param drop_columns_from_dataset:
    :return: ndarray of shape (n_samples, 2)
        Starting configuration of the projection result. By default it is ignored,
        and the starting projection is randomized using starting_projection_mode and random_state.
        If specified, this must match n_samples.
    """

    X = data

    if drop_columns_from_dataset is not None:
        X = np.delete(data, drop_columns_from_dataset, axis=1)

    return X


if __name__ == "__main__":

    projection_n_dimensions = 3
    # perplexity=85 works best for iris
    perplexity = 45.0
    max_iter = 300
    drop_columns_from_dataset = [-1]
    scaler = (0, 1)
    fix_column_to_z_projection_axis = -1
    plot = True

    # data = np.loadtxt('./datasets/mammals.data', delimiter=",")
    # data = pd.read_csv('./datasets/whr2019.csv', delimiter=",").values
    # data = np.concatenate((datasets.load_iris().data.T, [datasets.load_iris().target.T])).T
    data = np.concatenate((datasets.load_breast_cancer().data.T,[datasets.load_breast_cancer().target.T])).T
    # data = np.concatenate((datasets.load_boston().data.T,[datasets.load_boston().target.T])).T
    # data = np.tile(data, (100, 1)) # use this make the dataset 100x larger for performance test

    ################
    x = preprocess_data(data=data, drop_columns_from_dataset=drop_columns_from_dataset)
    dist_for_stress = create_triangular_distance_matrix(x)

    # normally perplexity should between 5-50,
    # but here for iris 85 let to smallest kruskal stress
    projection = tsne(data, no_dims=projection_n_dimensions, perplexity=perplexity, max_iter=max_iter,
                      fix_column_to_z_projection_axis=fix_column_to_z_projection_axis,
                      drop_columns_from_dataset=drop_columns_from_dataset,
                      scaler=scaler)

    stress = kruskal_stress(dist_for_stress, projection)
    print("kruskal stress is: ", stress)

    if plot:
        if projection_n_dimensions == 2:
            # show projection
            import plotly.graph_objects as go
            import numpy as np

            fig = go.Figure(
                data=go.Scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode='markers',
                    marker=dict(
                        size=16,
                        color=np.random.randn(500),
                        colorscale='Viridis',
                        showscale=True)
                )
            )
            fig.show()
        else:
            import plotly.graph_objects as go

            fig = go.Figure(
                data=[go.Scatter3d(x=projection[:, 0],
                                   y=projection[:, 1],
                                   z=projection[:, 2],
                                   mode='markers',
                                   marker=dict(
                                       size=12,
                                       color=projection[:, 2],  # set color to an array/list of desired values
                                       colorscale='Viridis',  # choose a colorscale
                                       opacity=0.8
                                   )
                                   )])

            fig.show()
