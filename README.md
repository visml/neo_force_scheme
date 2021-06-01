# NeoForceScheme

A new library for extended and performance-focused ForceScheme implementation.

## Installation
Until we set up a pypi package, you can test the library with

```
pip install git+https://github.com/visml/neo_force_scheme@0.0.1
```

## QuickStart

You can find examples for [simple data](./examples/mammals_cpu.py), [large data with cpu](./examples/mammals_large_cpu.py),
and [large data with gpu](./examples/mammals_large_gpu.py)

To run the projection:
```python
import numpy as np
from neo_force_scheme import NeoForceScheme

dataset = np.random.random((100, 100)) # Some dataset

nfs = NeoForceScheme()
projection = nfs.fit_transform(dataset)
```

To use GPU, be sure to have CUDA toolkit installed.
```python
import numpy as np
from neo_force_scheme import NeoForceScheme

dataset = np.random.random((100, 100)) # Some dataset

nfs = NeoForceScheme(cuda=True)
projection = nfs.fit_transform(dataset)
```

### Kruskal Stress
```python
import numpy as np
from neo_force_scheme import NeoForceScheme, kruskal_stress

dataset = np.random.random((100, 100)) # Some dataset

nfs = NeoForceScheme(cuda=True)
projection = nfs.fit_transform(dataset)

stress = kruskal_stress(nfs.embedding_, projection)
```

### Plot with matplotlib
```python
import matplotlib.pyplot as plt
import numpy as np
from neo_force_scheme import NeoForceScheme
from matplotlib.colors import ListedColormap

dataset = np.random.random((100, 100)) # Some dataset without labels
labels = np.random.random(100) # Per-row labels

nfs = NeoForceScheme(cuda=True)
projection = nfs.fit_transform(dataset)

plt.figure()
plt.scatter(projection[:, 0],
            projection[:, 1],
            c=labels,
            cmap=ListedColormap(['blue', 'red', 'green']),
            edgecolors='face',
            linewidths=0.5,
            s=4)
plt.grid(linestyle='dotted')
plt.show()
```

### Plot with plotly
```Python
import numpy as np
from neo_force_scheme import NeoForceScheme
import plotly.graph_objects as go

dataset = np.random.random((100, 100)) # Some dataset without labels
labels = np.random.random(100) # Per-row labels

nfs = NeoForceScheme(cuda=True)
projection = nfs.fit_transform(dataset)

fig = go.Figure(data=go.Scatter(x=projection[:, 0],
                                y=projection[:, 1],
                                mode='markers',
                                marker=dict(
                                size=16,
                                color=np.random.randn(100),
                                colorscale='Viridis',
                                showscale=True)
                                ))
fig.show()
```

## API
More information can be found at [our documentation page](#)