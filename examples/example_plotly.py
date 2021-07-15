from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn import datasets

from neo_force_scheme import NeoForceScheme, ProjectionMode

#################################
projection_n_dimensions = 3
nfs = NeoForceScheme(metric="euclidean", verbose=True, learning_rate0=0.5,
                     decay=0.95, max_it=100, cuda=False)
starting_projection_mode = ProjectionMode.RANDOM
plot = True

# data = np.loadtxt('./datasets/mammals.data', delimiter=",")
data = pd.read_csv('./datasets/whr2019.csv', delimiter=",").values
# data = np.concatenate((datasets.load_iris().data.T,[datasets.load_iris().target.T])).T
# data = np.concatenate((datasets.load_breast_cancer().data.T,[datasets.load_breast_cancer().target.T])).T
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

        fig = go.Figure(
            data=go.Scatter(
                x=projection[:, 0],
                y=projection[:, 1],
                mode='markers',
                marker=dict(
                    size=16,
                    color=np.random.randn(500) if t is None else t,
                    colorscale='Viridis',
                    showscale=True)
            )
        )
        fig.show()
    else:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[go.Scatter3d(x=projection[:, 0],
                               y=projection[:, 1],
                               z=projection[:, 2],
                               mode='markers',
                               marker=dict(
                                   size=12,
                                   color=np.random.randn(500) if t is None else t,  # set color to an array/list of desired values
                                   colorscale='Viridis',  # choose a colorscale
                                   opacity=0.8
                               )
                               )])

        fig.show()
