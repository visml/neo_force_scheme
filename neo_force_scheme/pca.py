from __future__ import print_function

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


def excute_pca(dataset,
               plot: Optional[bool] = False):
    rawdata = dataset

    size, dimension = rawdata.shape

    n, d = rawdata.shape

    x = rawdata[:, range(d - 1)]
    t = rawdata[:, d - 1]

    feat_cols = ['pixel' + str(i) for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feat_cols)
    df['y'] = t
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    # for debug purpose
    if plot == True:
        plt.figure()
        plt.scatter(pca_result[:, 0], pca_result[:, 1],
                    cmap=ListedColormap(['blue', 'red', 'green']), edgecolors='face', linewidths=0.5, s=4)
        plt.grid(linestyle='dotted')
        plt.show()

    return pca_result
