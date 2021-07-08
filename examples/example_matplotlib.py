from datetime import timedelta
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets

from neo_force_scheme import NeoForceScheme, ProjectionMode

#################################

projection_n_dimensions = 3
nfs = NeoForceScheme(metric="euclidean", verbose=True, learning_rate0=0.5,
                     decay=0.95, max_it=500, cuda=False)
starting_projection_mode = ProjectionMode.RANDOM
plot = True

data = np.loadtxt('./mammals.data', delimiter=",")
# data = datasets.load_iris().data
# data = datasets.load_breast_cancer().data
# data = np.tile(data, (100, 1)) # use this make the dataset 100x larger for performance test

#################################

n, d = data.shape[0], data.shape[1]
x = data[:, range(d - 1)]
t = data[:, d - 1]


# execute force scheme
start = timer()
projection = nfs.fit_transform(X=x, Xd=data,
                               starting_projection_mode=starting_projection_mode,
                               random_state=1,
                               n_dimension=projection_n_dimensions)
error = nfs.projection_error_
end = timer()

print(f'ForceScheme took {timedelta(seconds=end - start)} to execute with error {error}')

# calculate stress
stress = nfs.score(projection)
print('Kruskal stress {0}:'.format(stress))

# save projection
# np.savetxt(input_file + "_projection.txt", projection, delimiter=" ", fmt="%s")

if plot:
    if projection_n_dimensions == 2:
        # show projection
        plt.figure()
        plt.scatter(projection[:, 0],
                    projection[:, 1],
                    c=t,
                    cmap=ListedColormap(['blue', 'red', 'green']),
                    edgecolors='face',
                    linewidths=0.5,
                    s=4)
        plt.grid(linestyle='dotted')
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(projection[:, 0], projection[:, 1], projection[:, 2], c=t)
        plt.show()