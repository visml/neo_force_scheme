from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def excute_tsne(dataset,
                plot: Optional[bool] = False,
                n_dimension: Optional[int] = 2):
    dataset_embedded = TSNE(n_components=n_dimension).fit_transform(dataset)
    return dataset_embedded

def excute_pca(dataset,
               plot: Optional[bool] = False,
               n_dimension: Optional[int] = 2):
    dataset_embedded = PCA(n_components=n_dimension).fit_transform(dataset)
    return dataset_embedded