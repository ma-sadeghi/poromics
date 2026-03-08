# Tortuosity (FD solver)

## Background

The FD solver computes tortuosity by solving the steady-state diffusion equation on the pore space:

$$\nabla \cdot (D \, \nabla c) = 0$$

where $c$ is the concentration field and $D$ is the diffusivity (scalar or spatially variable). Dirichlet boundary conditions fix the concentration on two opposing faces ($c = 1$ at the inlet, $c = 0$ at the outlet), with periodic conditions on the remaining faces. Solid voxels act as no-flux boundaries.

The resulting linear system is solved using a Krylov method (conjugate gradient). Tortuosity is then extracted from the concentration field:

$$\tau = \frac{\varepsilon}{D_\text{eff} / D_0}$$

where $\varepsilon$ is the porosity, $D_\text{eff}$ is the effective diffusivity obtained from the steady-state flux, and $D_0$ is the bulk diffusivity.

This solver is implemented in Julia via [Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/) and supports GPU acceleration through CUDA.jl.

!!! note

    The first call triggers a one-time Julia installation (~2-3 minutes). Subsequent calls reuse a persistent Julia subprocess, so JIT compilation is cached.

## Usage

```python
# mkdocs: render
# mkdocs: hidecode
# mkdocs: hideoutput
import poromics
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, blobiness=0.5, seed=42)
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
```

```python exec="1" session="tau-fd"
# mkdocs: hidecode
import poromics
import porespy as ps

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, blobiness=0.5, seed=42)
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
```

```python exec="1" session="tau-fd"
print(f"Tortuosity: {result.tau:.4f}")
print(f"Effective diffusivity (D_eff/D_0): {result.D_eff:.6f}")
print(f"Formation factor: {result.formation_factor:.4f}")
```

### Spatially variable diffusivity

The FD solver accepts an ndarray for `D`, enabling simulations where diffusivity varies across the domain (e.g., composite electrodes with different phases):

```python
import numpy as np

D = np.ones_like(im, dtype=float)
D[im] = 1.0    # pore phase
D[~im] = 0.0   # solid phase (no diffusion)
result = poromics.tortuosity_fd(im, axis=1, D=D, rtol=1e-5)
```

## Result

`tortuosity_fd` returns a `TortuosityResult`:

| Attribute          | Description                                    |
|--------------------|------------------------------------------------|
| `im`               | Boolean image used in the simulation           |
| `axis`             | Axis along which the simulation was run        |
| `porosity`         | Pore volume fraction                           |
| `tau`              | Tortuosity factor ($\geq 1$)                   |
| `D_eff`            | Normalized effective diffusivity ($D_\text{eff} / D_0$) |
| `c`                | Steady-state concentration field               |
| `formation_factor` | Formation factor $F = 1 / D_\text{eff}$        |
| `D`                | Bulk diffusivity (float or ndarray)            |

## Visualize the concentration field

```python
# mkdocs: render
# mkdocs: hidecode
fig, ax = plt.subplots()
cax = ax.imshow(result.c[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Concentration Field")
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("c")
```
