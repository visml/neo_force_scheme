import numpy as np


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
