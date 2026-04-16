# Quick start

After installing `poromics`, you can estimate transport properties in just a few lines. All solvers accept 2D or 3D binary images where `True` = pore and `False` = solid.

!!! note

    The first time you call `tortuosity_fd`, Julia and its dependencies are installed automatically (one-time setup). The LBM solvers (`tortuosity_lbm`, `permeability_lbm`) use Taichi and work out of the box.

## Generate a test image

```python
# mkdocs: render
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, blobiness=0.5, seed=42)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(im[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Boolean Image")
```

```python exec="1" session="quickstart"
# mkdocs: hidecode
import poromics
import porespy as ps

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, blobiness=0.5, seed=42)
```

## Tortuosity (FD)

```python exec="1" source="above" session="quickstart"
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
print(result)
```

## Tortuosity (LBM)

```python exec="1" source="above" session="quickstart"
result = poromics.tortuosity_lbm(im, axis=1, D=1e-9, voxel_size=1e-6)
print(result)
```

## Permeability

```python exec="1" source="above" session="quickstart"
result = poromics.permeability_lbm(im, axis=1, nu=1e-6, voxel_size=1e-6)
print(result)
```

## What's next?

- Learn about each solver's physics and options: [Solvers](solvers/tortuosity-fd.md)
- See worked examples with visualization: [Examples](examples/index.md)
- Use the lower-level simulation API for custom workflows: [Simulation API](simulation.md)
