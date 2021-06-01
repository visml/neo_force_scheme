from datetime import timedelta
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from neo_force_scheme import NeoForceScheme, kruskal_stress

data = np.loadtxt("./mammals.data", delimiter=",")
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
"""
# method1
import plotly.express as px
fig = px.scatter(x=projection[:, 0],
                 y=projection[:, 1],
                 labels={'x':'x', 'y':'y'})
fig.show()
"""
"""
# method2
import numpy as np
import plotly.express as px

ar = np.arange(100).reshape((10, 10))
fig = px.scatter(projection,
                 x=projection[:, 0],
                 y=projection[:, 1],
                 size=1,
                 color=1)
fig.show()
"""

# method3
import plotly.graph_objects as go
import numpy as np

fig = go.Figure(data=
                go.Scatter(x=projection[:, 0],
                            y=projection[:, 1],
                            mode='markers',
                            marker=dict(
                                size=16,
                                color=np.random.randn(500),
                                colorscale='Viridis',
                                showscale=True)
                        )
                )
fig.show()

