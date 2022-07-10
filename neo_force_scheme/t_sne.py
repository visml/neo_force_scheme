# coding:utf-8
from typing import Optional, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
import time
from .engines import new_technique
from .engines import neo_force_scheme_cpu
from sklearn.base import BaseEstimator
from . import distances
"""
     Copyright 2020 heucoder
     @ https://github.com/heucoder/dimensionality_reduction_alo_codes/blob/master/codes/T-SNE/TSNE.py

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 Access date: Aug 2021

 Changes made from the original code:
 1. Translated the Chinese comments into English for developing convenience.
 2. Added parameters and relative features in tsne() to adapt
    the fixed axis technique.
 3. Added functions which will be used by fixed axis technique 
"""
'''
The file uses t-SNE to illustrate how to use TECHNIQUE based on different methods.
Different from the main neo-force-scheme class, this class has fewer options. 
No move-in-range option and distance metrics option are available.
Only fixed-axis option is available with eucliean distance.
'''
class TSNE(BaseEstimator):
    def __init__(
            self,
            *,
            metric_args: list = None,
            max_it: int = 100,
            learning_rate0: float = 0.5,
            decay: float = 0.95,
            tolerance: float = 0.00001,
            n_jobs: int = None,
            cuda: bool = False,
            cuda_threads_per_block: Optional[int] = None,
            cuda_blocks_per_grid: Optional[int] = None,
            cuda_profile: bool = False,
            verbose: bool = False,
    ):
        self.metric_args = metric_args
        self.max_it = max_it
        self.learning_rate0 = learning_rate0
        self.decay = decay
        self.tolerance = tolerance
        self.n_jobs = n_jobs
        self.cuda = cuda
        self.cuda_profile = cuda_profile
        self.cuda_threads_per_block = cuda_threads_per_block
        self.cuda_blocks_per_grid = cuda_blocks_per_grid
        self.print = print if verbose else lambda *a, **k: None

    def cal_pairwise_dist(self, x):
        '''Calculate the distance of a pairwise, x is a matrix
        (a-b)^2 = a^2 + b^2 - 2*a*b
        '''
        sum_x = np.sum(np.square(x), 1)
        dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
        # 返回任意两个点之间距离的平方
        return dist

    def cal_perplexity(self, dist, idx=0, beta=1.0):
        '''Calculate perplexity. D is distance vector
        idx is the distance between a point and itself,
        beta is Gaussian distribution parameter
        '''
        prob = np.exp(-dist * beta)
        # set a point's own prob to 0
        prob[idx] = 0
        sum_prob = np.sum(prob)
        if sum_prob < 1e-12:
            prob = np.maximum(prob, 1e-12)
            perp = -12
        else:
            perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
            prob /= sum_prob

        return perp, prob

    def seach_prob(self, x, tol=1e-5, perplexity=30.0):
        '''Using binary research to find beta,
        then calculate the prob of the pairwise
        '''

        # Initialize parameters
        print("Computing pairwise distances...")
        (n, d) = x.shape
        dist = self.cal_pairwise_dist(x)
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
            perp, this_prob = self.cal_perplexity(dist[i], i, beta[i])

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

                # update perb,prob
                perp, this_prob = self.cal_perplexity(dist[i], i, beta[i])
                perp_diff = perp - base_perp
                tries = tries + 1
            # record prob value
            pair_prob[i,] = this_prob
        print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
        return pair_prob

    def tsne(self, x, no_dims=2, perplexity=30.0, max_iter=1000,
             fix_column_to_z_projection_axis: Optional[int] = None,
             drop_columns_from_dataset: Optional[List[int]] = None,
             scaler: Optional[Tuple[int, int]] = (0, 1)):
        """Runs t-SNE on the dataset in the NxD array x
        to reduce its dimensionality to no_dims dimensions.
        The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
        where x is an NxD NumPy array.
        """

        # Check inputs
        if isinstance(no_dims, float):
            print("Error: array x should have type float.")
            return -1

        if scaler is not None:
            x = neo_force_scheme_cpu.scale_dataset(x, scaler)

        z_axis_fixed = x[:, fix_column_to_z_projection_axis]
        x = new_technique.preprocess_data(data=x, drop_columns_from_dataset=drop_columns_from_dataset)

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
        P = self.seach_prob(x, 1e-5, perplexity)
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
            y = y + iy
            y = y - np.tile(np.mean(y, 0), (n, 1))
            # Compute current value of cost function\
            if (iter + 1) % 100 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration ", (iter + 1), ": error is ", C)
                if (iter + 1) != 100:
                    ratio = C / oldC
                    print("ratio ", ratio)
                    if ratio >= 0.95:
                        break
                oldC = C
            # Stop lying about P-values
            if iter == 100:
                P = P / 4
        print("finished training!")
        return y

    # TODO: add evaluating method.
    def score(self, dataset, projection, drop_columns_from_dataset, scaler=(0, 1)):
        """Calculates the kruskal stress of the projection.
        Uses the calculated distance matrix by default, but can be given a custom one if needed
        Parameters
        ----------
        projection : ndarray of shape (n_samples, 2)
            Result of the transform operation (aka the resulting projection)
        distance_matrix : Optional custom distance matrix to calculate the score from
        Returns
        -------
        score: the kruskal_stress between 0 and 1. Represents how well the projection represents the original distances.
        Numbers below 0.1 are considered low,
        between 0.1 and 0.3 medium, between 0.3 and 0.5 high, and above 0.5 very high.
        """
        dataset = new_technique.preprocess_data(data=dataset, drop_columns_from_dataset=drop_columns_from_dataset)

        distance_matrix = neo_force_scheme_cpu.create_triangular_distance_matrix(dataset, distances.euclidean)
        ret = np.copy(projection)
        if scaler is not None:
            ret = neo_force_scheme_cpu.scale_dataset(ret, scaler)

        return neo_force_scheme_cpu.kruskal_stress(distance_matrix, ret, distances.euclidean)

