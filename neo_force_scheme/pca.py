from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
from matplotlib.colors import ListedColormap
from sklearn import datasets

def excute_pca(dataset):
    print("Importing data.")
    rawdata = dataset
    print("Done.")

    size, dimension = rawdata.shape

    n, d = rawdata.shape

    x = rawdata[:, range(d - 1)]
    t = rawdata[:, d - 1]

    feat_cols = ['pixel' + str(i) for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feat_cols)
    df['y'] = t
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    """ for debug purpose
    plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1],
                cmap=ListedColormap(['blue', 'red', 'green']), edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()
    """
    return pca_result
