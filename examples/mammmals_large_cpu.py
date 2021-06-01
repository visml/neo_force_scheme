from datetime import timedelta
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from neo_force_scheme import NeoForceScheme, kruskal_stress

data = np.loadtxt("./mammals.data", delimiter=",")
data = np.tile(data, (100, 1))
n, d = data.shape[0], data.shape[1]
x = data[:, range(d - 1)]
t = data[:, d - 1]

# read the distance matrix
nfs = NeoForceScheme(verbose=True, learning_rate0=0.2, decay=0.95)

# execute force scheme
start = timer()
projection = nfs.fit_transform(x, random_state=1)
error = nfs.projection_error_
end = timer()

print(f'ForceScheme took {timedelta(seconds=end - start)} to execute with error {error}')

# calculate stress
stress = kruskal_stress(nfs.embedding_, projection)
print('Kruskal stress {0}:'.format(stress))

# save projection
# np.savetxt(input_file + "_projection.txt", projection, delimiter=" ", fmt="%s")

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
