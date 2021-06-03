import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import datasets

def excute_tsne(dataset):
    print("Importing data.")
    rawdata = dataset
    print("Done.")

    size, dimension = rawdata.shape

    x = rawdata[:, range(dimension - 1)]
    t = rawdata[:, dimension - 1]

    feat_cols = ['pixel' + str(i) for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feat_cols)
    df['y'] = t
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))

    # excute tsne
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(df[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    """ for debug purpose
    plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
              cmap=ListedColormap(['blue', 'red', 'green']), edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()
    """
    return tsne_result
