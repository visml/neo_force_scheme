# NeoForceScheme
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4891820.svg)](https://doi.org/10.5281/zenodo.4891820)

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

## API
More information can be found at [our documentation page](#)

# CITATION
```bibtex
@misc{christino_paulovich_4891820,
  author       = {Paulovich, Fernando and Christino, Leonardo},
  title        = {NeoForceScheme},
  version      = {0.0.2},
  publisher    = {Zenodo},
  month        = may,
  year         = 2021,
  doi          = {10.5281/zenodo.4891820},
  url          = {http://dx.doi.org/10.5281/zenodo.4891820}
    
}
```