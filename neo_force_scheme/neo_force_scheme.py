import math
import traceback
from datetime import timedelta
from enum import Enum
from sys import getsizeof
from timeit import default_timer as timer
from typing import Optional, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from . import distances
from .engines import neo_force_scheme_cpu
from .engines.neo_force_scheme_cpu import scale_dataset
from .engines.new_technique import preprocess_data, get_z_score


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
            millions of examples_old.
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
            neo_force_scheme_cpu.pickle_save_matrix(filename, self.embedding_, self.embedding_size_)
        else:
            raise NotImplemented('Only pickle save method is currently implemented')

    def load(self, filename, *, use_pickle=True):
        if use_pickle:
            self.embedding_, self.embedding_size_ = neo_force_scheme_cpu.pickle_load_matrix(filename)
        else:
            self.embedding_, self.embedding_size_ = neo_force_scheme_cpu.read_distance_matrix(filename)

    def _fit(self, X,
             drop_columns_from_dataset):
        X = preprocess_data(data=X, drop_columns_from_dataset=drop_columns_from_dataset)

        self.embedding_ = neo_force_scheme_cpu.create_triangular_distance_matrix(X, self.metric)
        self.print(f'Distance matrix size in memory: ', round(getsizeof(self.embedding_) / 1024 / 1024, 2), 'MB')


    def _transform(self, X, *, index, total, inplace, n_dimension: Optional[int] = 2, force_projection_dimensions=None,
                   original_z_axis=None, z_axis_moving_range: Optional[Tuple[float, float]] = (0, 0),
                   confidence_interval: Optional[float] = 1):
        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf

        z_score = 1
        if confidence_interval < 1.0:
            range_strict_limitation = False
            get_z_score(confidence_interval)
            # z_score is calculated separately because it is using scipy function
            # which is not compatible with numba
        else:
            range_strict_limitation = True
        for k in range(self.max_it):
            learning_rate = self.learning_rate0 * math.pow((1 - k / self.max_it), self.decay)
            new_error = neo_force_scheme_cpu.iteration(index=index,
                                                       distance_matrix=self.embedding_,
                                                       projection=X,
                                                       learning_rate=learning_rate,
                                                       n_dimension=n_dimension,
                                                       metric=self.metric,
                                                       force_projection_dimensions=force_projection_dimensions,
                                                       original_z_axis=original_z_axis,
                                                       z_axis_moving_range=z_axis_moving_range,
                                                       range_strict_limitation=range_strict_limitation,
                                                       z_score=z_score)
                                                       # z_score is passed in all situations,
                                                       # but will only be used when range_strict_limitation == False

            if math.fabs(new_error - error) < self.tolerance:
                self.print(f'Error below tolerance {math.fabs(new_error - error)} in iteration {k}, breaking')
                break

            error = new_error
        self.print(f'Max iteration reached, breaking!')
        return X, error

    def transform(
            self,
            Xd: Optional[np.array] = None,
            *,
            starting_projection_mode: Optional[ProjectionMode] = ProjectionMode.RANDOM,
            inplace: bool = True,  # TODO: implement False
            random_state: float = None,
            n_dimension: Optional[int] = 2,
            fix_column_to_z_projection_axis=None,
            z_axis_moving_range: Optional[Tuple[float, float]] = (0, 0),
            confidence_interval: Optional[float] = 0.99,
            X=None,
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
        inplace: boolean
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
            self.print(f'Using custom random state: {random_state}')

        index = np.random.permutation(size)

        if starting_projection_mode is not None:
            # randomly initialize the projection
            if starting_projection_mode == ProjectionMode.RANDOM or Xd is None:
                Xd = np.random.random((size, n_dimension))
            # initialize the projection with tsne
            if starting_projection_mode == ProjectionMode.TSNE:
                # TODO: Allow user input for tsne iteration time.
                # Note: bigger the iteration time, larger the final kruskal stress.
                Xd = TSNE(n_components=n_dimension, n_iter=self.max_it, n_jobs=self.n_jobs,
                          random_state=random_state).fit_transform(Xd)
            # initialize the projection with pca
            elif starting_projection_mode == ProjectionMode.PCA:
                Xd = PCA(n_components=n_dimension, random_state=random_state).fit_transform(Xd)

        elif Xd is None:
            raise Exception('Either Xd needs to be provided or a starting_projection_mode needs to be chosen')

        # Manually set z axis to be a certain feature
        force_projection_dimensions = None
        if fix_column_to_z_projection_axis is not None:
            force_projection_dimensions = np.arange(n_dimension - 1)
            Xd[:, n_dimension-1] = X[:, fix_column_to_z_projection_axis]

        original_z_axis = np.copy(Xd[:, n_dimension-1])

        if n_dimension > 3:
            raise NotImplementedError('projection for a dimension bigger than 3 is not implemented yet!')

        if self.cuda:
            Xd, self.projection_error_ = self._gpu_transform(Xd, index=index, total=total, inplace=inplace,
                                                             n_dimension=n_dimension)
        else:
            Xd, self.projection_error_ = self._transform(Xd, index=index, total=total, inplace=inplace,
                                                         n_dimension=n_dimension,
                                                         force_projection_dimensions=force_projection_dimensions,
                                                         original_z_axis=original_z_axis,
                                                         z_axis_moving_range=z_axis_moving_range,
                                                         confidence_interval=confidence_interval)

        Xmin = Xd.min(axis=0)
        Xd = Xd - Xmin

        return Xd

    def _gpu_transform(self, X, *, index, total, inplace, n_dimension):
        if n_dimension > 3:
            raise NotImplementedError('4d version for gpu is not implemented yet!')

        try:
            from .engines.neo_force_scheme_gpu import gpu_transform
            return gpu_transform(self, X=X, index=index, total=total, inplace=inplace)
        except Exception as e:
            print(f'Unable to use GPU due to exception {e}. Defaulting to CPU')
            traceback.print_stack()
            return self._transform(X, index=index, total=total, inplace=inplace)


    def fit_transform(self,
                      X: np.array,
                      *,
                      Xd: Optional[np.array] = None,
                      fix_column_to_z_projection_axis: Optional[int] = None,
                      drop_columns_from_dataset: Optional[List[int]] = None,
                      scaler: Optional[Tuple[int, int]] = (0, 1),
                      z_axis_moving_range: Optional[Tuple[float, float]] = (0, 0),
                      confidence_interval: Optional[float] = 1,
                      **kwargs):
        """
        Fit X into an embedded space and return that transformed
        output.
        :param X: dataset as ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        :param Xd: optional ndarray of shape (n_samples, 2)
            Starting configuration of the projection result. By default it is ignored,
            and the starting projection is randomized using starting_projection_mode and random_state.
            If specified, this must match n_samples.
        :param fix_column_to_z_projection_axis: indicate the column which is used as z axis in 3d version
            or y axis in 2d version of the ploted figure.
        :param drop_columns_from_dataset: indicate the columns that you do not want to be
            used for calculating distance matrix. Default is None.
        :param scaler: the range to which all columns of the dataset should be scaled to.
            Default is (0, 1). If the user does not want the dataset to be scaled, the input
            should be None.
        :param z_axis_moving_range: the range to which points are allowed to move on z axis.
            Default is (0, 0). This parameter will only be used when fix_column_to_z_projection_axis
            is not None.
        :param confidence_interval: indicate the parameter which will be used to calculating the actual
            moving distance when z_axis_moving_range is not (0, 0). Default is 1, which means
            moving range is followed strictly. Otherwise the moving distance will be calculated
            using a Gaussian function.
        :param kwargs:
            starting_projection_mode: one of [RANDOM], [PCA], [TSNE]
                Specifies the starting values of the projection.
                Utilize if X is None
            inpalce: boolean
                Specifies whether X will be changed inplace during ForceScheme projection
            random_state: float
                Specifies the starting random state used for randomization
        :return: ndarray of shape (n_samples, 2)
            Embedding of the training data in low-dimensional space.
        """
        start = timer()

        # TODO: calculate moving range automatically based on the range of normalized z axis
        if scaler is not None:
            X = scale_dataset(X, scaler)

        self._fit(X, drop_columns_from_dataset=drop_columns_from_dataset)
        ret = self.transform(Xd, fix_column_to_z_projection_axis=fix_column_to_z_projection_axis,
                             z_axis_moving_range=z_axis_moving_range, X=X,
                             confidence_interval=confidence_interval, **kwargs)
        end = timer()
        self.print(f'Time elapsed: {timedelta(seconds=end - start)}')
        return ret

    def fit(self, X, y=None,
            drop_columns_from_dataset=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        y : Ignored
        """
        self._fit(X,
                  drop_columns_from_dataset)
        return self

    def score(self, projection, *, distance_matrix: Optional[np.array] = None, scaler=(0, 1)):
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
        if distance_matrix is None:
            distance_matrix = self.embedding_
        if distance_matrix is None:
            raise Exception(
                'Please run a transform operation or provide a custom distance matrix before calling the score')
        ret = np.copy(projection)
        if scaler is not None:
            ret = scale_dataset(ret, scaler)
        return neo_force_scheme_cpu.kruskal_stress(self.embedding_, ret, self.metric)

