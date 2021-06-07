from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import plotly.graph_objects as go

from neo_force_scheme import NeoForceScheme, ProjectionMode, kruskal_stress

data = np.loadtxt("./mammals.data", delimiter=",")
n, d = data.shape[0], data.shape[1]
x = data[:, range(d - 1)]
t = data[:, d - 1]

# read the distance matrix
nfs = NeoForceScheme(verbose=True, learning_rate0=0.2, decay=0.95)

# execute force scheme
start = timer()
# projection = nfs.fit_transform(X=x, starting_projection_mode=ProjectionMode.RANDOM, random_state=1)
projection = nfs.fit_transform(X=x, Xd=data, starting_projection_mode=ProjectionMode.TSNE, random_state=1)
# projection = nfs.fit_transform(X=x, Xd=data, starting_projection_mode=ProjectionMode.PCA, random_state=1)

error = nfs.projection_error_
end = timer()

print(f'ForceScheme took {timedelta(seconds=end - start)} to execute with error {error}')

# calculate stress
stress = kruskal_stress(nfs.embedding_, projection)
print('Kruskal stress {0}:'.format(stress))

# save projection
# np.savetxt(input_file + "_projection.txt", projection, delimiter=" ", fmt="%s")

# show projection
fig = go.Figure(
    data=[go.Scatter3d(x=projection[:, 0],
                       y=projection[:, 1],
                       z=projection[:, 2],
                       mode='markers',
                       marker=dict(
                           size=12,
                           color=projection[:, 2],  # set color to an array/list of desired values
                           colorscale='Viridis',  # choose a colorscale
                           opacity=0.8
                       )
                       )])

fig.show()
