import math
import traceback
from enum import Enum
from sys import getsizeof
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from . import utils, distances, tsne, pca

TPB = 16


class ProjectionMode(Enum):
    RANDOM = 1
    TSNE = 2
    PCA = 3


class NeoForceScheme(BaseEstimator):
    """ForceScheme Projection technique.

    Force Scheme is???

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

        Parameters
        ----------
        learning_rate : float, default=200.0
            The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
            the learning rate is too high, the data may look like a 'ball' with any
            point approximately equidistant from its nearest neighbours. If the
            learning rate is too low, most points may look compressed in a dense
            cloud with few outliers. If the cost function gets stuck in a bad local
            minimum increasing the learning rate may help.
        n_iter : int, default=1000
            Maximum number of iterations for the optimization. Should be at
            least 250.
        n_iter_without_progress : int, default=300
            Maximum number of iterations without progress before we abort the
            optimization, used after 250 initial iterations with early
            exaggeration. Note that progress is only checked every 50 iterations so
            this value is rounded to the next multiple of 50.
            .. versionadded:: 0.17
               parameter *n_iter_without_progress* to control stopping criteria.
        min_grad_norm : float, default=1e-7
            If the gradient norm is below this threshold, the optimization will
            be stopped.
        metric : str or callable, default='euclidean'
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string, it must be one of the options
            allowed by scipy.spatial.distance.pdist for its metric parameter, or
            a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
            If metric is "precomputed", X is assumed to be a distance matrix.
            Alternatively, if metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays from X as input and return a value indicating
            the distance between them. The default is "euclidean" which is
            interpreted as squared euclidean distance.
        init : {'random', 'pca'} or ndarray of shape (n_samples, n_components), \
                default='random'
            Initialization of embedding. Possible options are 'random', 'pca',
            and a numpy array of shape (n_samples, n_components).
            PCA initialization cannot be used with precomputed distances and is
            usually more globally stable than random initialization.
        verbose : int, default=0
            Verbosity level.
        random_state : int, RandomState instance or None, default=None
            Determines the random number generator. Pass an int for reproducible
            results across multiple function calls. Note that different
            initializations might result in different local minima of the cost
            function. See :term: `Glossary <random_state>`.
        method : str, default='barnes_hut'
            By default the gradient calculation algorithm uses Barnes-Hut
            approximation running in O(NlogN) time. method='exact'
            will run on the slower, but exact, algorithm in O(N^2) time. The
            exact algorithm should be used when nearest-neighbor errors need
            to be better than 3%. However, the exact method cannot scale to
            millions of examples.
            .. versionadded:: 0.17
               Approximate optimization *method* via the Barnes-Hut.
        Attributes
        ----------
        embedding_ : array-like of shape (n_samples, n_components)
            Stores the embedding vectors.
        kl_divergence_ : float
            Kullback-Leibler divergence after optimization.
        n_iter_ : int
            Number of iterations run.
        Examples
        --------
        # >>> import numpy as np
        # >>> from sklearn.manifold import TSNE
        # >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        # >>> X_embedded = TSNE(n_components=2).fit_transform(X)
        # >>> X_embedded.shape
        (4, 2)
        References
        ----------
        [1] ...
        """

    def __init__(
            self,
            *,
            metric="euclidean",
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
        try:
            self.metric = getattr(distances, metric)
        except Exception as e:
            raise NotImplemented(f'Metric {metric} is not implemented', e)

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

    def save(self, filename, *, use_pickle=True):
        if use_pickle:
            utils.pickle_save_matrix(filename, self.embedding_, self.embedding_size_)
        else:
            raise NotImplemented('Only pickle save method is currently implemented')

    def load(self, filename, *, use_pickle=True):
        if use_pickle:
            self.embedding_, self.embedding_size_ = utils.pickle_load_matrix(filename)
        else:
            self.embedding_, self.embedding_size_ = utils.read_distance_matrix(filename)

    def _fit(self, X, skip_num_points=0):
        self.embedding_ = utils.create_triangular_distance_matrix(X, self.metric)
        self.print(f'Distance matrix size in memory: ', round(getsizeof(self.embedding_) / 1024 / 1024, 2), 'MB')

    def _transform(self, X, *, index, total, inplace):
        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        for k in range(self.max_it):
            learning_rate = self.learning_rate0 * math.pow((1 - k / self.max_it), self.decay)
            new_error = utils.iteration(index, self.embedding_, X, learning_rate)

            if math.fabs(new_error - error) < self.tolerance:
                break

            error = new_error
        return X, error

    def transform(
            self,
            X: Optional[np.array] = None,
            Xd: Optional[np.array] = None,
            *,
            starting_projection_mode: Optional[ProjectionMode] = ProjectionMode.RANDOM,
            inpalce: bool = True,  # TODO: implement False
            random_state: float = None,
    ):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        Xd : array, shape (n_samples, n_features)
            New data to be transformed.
        starting_projection_mode: one of [RANDOM]
            Specifies the starting values of the projection.
            Utilize if X is None
        inpalce: boolean
            Specifies whether X will be changed inplace during ForceScheme projection

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        check_is_fitted(self)
        total = len(self.embedding_)
        size = int(math.sqrt(2 * total + 1))

        # set the random seed
        if random_state is not None:
            np.random.seed(random_state)

        if starting_projection_mode is not None:
            # randomly initialize the projection
            if starting_projection_mode == ProjectionMode.RANDOM:
                #print(starting_projection_mode)
                Xd = np.random.random((size, 2))
            # initialize the projection with tsne
            else:
                if starting_projection_mode == ProjectionMode.TSNE:
                    Xd = tsne.excute_tsne(X)
            # initialize the projection with pca
                else:
                    if starting_projection_mode == ProjectionMode.PCA:
                        Xd = pca.excute_pca(X)
        # create random index TODO: other than random
        index = np.random.permutation(size)

        if self.cuda:
            Xd, self.projection_error_ = self._gpu_transform(Xd, index=index, total=total, inplace=inpalce)
        else:
            Xd, self.projection_error_ = self._transform(Xd, index=index, total=total, inplace=inpalce)

        # setting the min to (0,0)
        min_x = min(Xd[:, 0])
        min_y = min(Xd[:, 1])
        for i in range(size):
            Xd[i][0] -= min_x
            Xd[i][1] -= min_y

        return Xd

    def _gpu_transform(self, X, *, index, total, inplace):
        try:
            from numba import cuda
            from .cuda_utils import calc_error, iteration
            # iterate until max_it or if the error does not change more than the tolerance
            error = math.inf

            d_distance_matrix = cuda.to_device(self.embedding_)
            d_index = cuda.to_device(index)
            d_projection = cuda.to_device(X)

            size = int(math.sqrt(2 * total + 1))
            threadsperblock = self.cuda_threads_per_block if self.cuda_threads_per_block else (TPB, TPB)
            if self.cuda_blocks_per_grid is not None:
                blockspergrid = self.cuda_blocks_per_grid
            elif isinstance(threadsperblock, int):
                blockspergrid = (size) // threadsperblock
            else:
                blockspergrid = ((size) // threadsperblock[0], (size) // threadsperblock[1])
            self.print(f'Dist: threads:{threadsperblock}, blocks:{blockspergrid}')

            if self.cuda_profile:
                cuda.profile_start()

            ref_new_error = np.zeros(shape=(size, size), dtype=np.float64)
            d_new_error = cuda.to_device(ref_new_error)
            for k in range(self.max_it):
                learning_rate = self.learning_rate0 * math.pow((1 - k / self.max_it), self.decay)
                iteration[blockspergrid, threadsperblock](d_index, d_distance_matrix, d_projection, learning_rate, d_new_error)
                new_error = calc_error(d_new_error.reshape(size * size)) / (size)
                # d_new_error.copy_to_host(ref_new_error)
                # new_error = ref_new_error.sum()/size
                # self.print(new_error, error, size, np.isinf(ref_new_error).sum(), np.isnan(ref_new_error).sum())
                if math.fabs(new_error - error) < self.tolerance:
                    break

                error = new_error

            if self.cuda_profile:
                cuda.profile_stop()
            self.print(f'Iter: {k} error: {error}')

            if inplace:
                d_projection.copy_to_host(X)
                return X, error
            else:
                tmp = None
                d_projection.copy_to_host(tmp)
                return tmp, error
        except Exception as e:
            print(f'Unable to use GPU due to exception {e}. Defaulting to CPU')
            traceback.print_stack()
            return self._transform(X, index=index, total=total, inplace=inplace)

    def fit_transform(self, X, Xd=None, **kwargs):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        Xd : ndarray of shape (n_samples, 2)
            Starting configuration of the projection result. By default it is ignored,
            and the starting projection is randomized using starting_projection_mode and random_state.
            If specified, this must match n_samples.
        starting_projection_mode: one of [RANDOM], [PCA], [TSNE]
            Specifies the starting values of the projection.
            Utilize if X is None
        inpalce: boolean
            Specifies whether X will be changed inplace during ForceScheme projection
        random_state: float
            Specifies the starting random state used for randomization

        Returns
        -------
        X_new : ndarray of shape (n_samples, 2)
            Embedding of the training data in low-dimensional space.
        """
        self._fit(X)
        return self.transform(Xd, **kwargs)

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        y : Ignored
        """
        self._fit(X)
        return self
