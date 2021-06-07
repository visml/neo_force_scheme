from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE


def excute_tsne(dataset,
                plot: Optional[bool] = False,
                n_dimension: Optional[int] = 2):
    rawdata = dataset

    size, dimension = rawdata.shape

    x = rawdata[:, range(dimension - 1)]
    t = rawdata[:, dimension - 1]

    feat_cols = ['pixel' + str(i) for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feat_cols)
    df['y'] = t
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None

    # excute tsne
    tsne = TSNE(n_components=n_dimension, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(df[feat_cols].values)

    # for debug purpose
    if plot == True:
        plt.figure()
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                    cmap=ListedColormap(['blue', 'red', 'green']), edgecolors='face', linewidths=0.5, s=4)
        plt.grid(linestyle='dotted')
        plt.show()

    return tsne_result
