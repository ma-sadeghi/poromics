# Basic usage

After installing `poromics`, you can use it in your Python scripts or Jupyter notebooks. Poromics provides three main functions for estimating transport properties, plus lower-level simulation solvers for more control.

!!! note

    The first time you call `tortuosity_fd`, it will take a few minutes to install Julia and the required packages. This is a one-time setup. The LBM solvers (`tortuosity_lbm`, `permeability_lbm`) use Taichi and do not require Julia.

## Generate a test image

You can use the `porespy` package to generate a test image. In this example, we will create a simple 2D image with `blobs`.

```python
# mkdocs: render
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, seed=42)

fig, ax = plt.subplots()
ax.imshow(im[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Boolean Image")
```

## Tortuosity (Julia FD solver)

The Julia-based FD solver uses Krylov iterations to solve the steady-state diffusion equation. It supports spatially variable diffusivity fields.

```python
import poromics

result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=True)
print(result)
```

```python exec="1"
# mkdocs: hidecode
import poromics
import porespy as ps

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, seed=42)
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
print(result)
```

## Tortuosity (LBM solver)

The Taichi-based LBM solver uses the D3Q7 BGK lattice Boltzmann method. It accepts physical units for diffusivity and voxel size.

```python
import poromics

result = poromics.tortuosity_lbm(im, axis=1, D=1e-9, voxel_size=1e-6)
print(result.tau, result.D_eff)
```

## Permeability (LBM solver)

The permeability solver uses the D3Q19 MRT lattice Boltzmann method to solve creeping (Stokes) flow and extract permeability via Darcy's law.

```python
import numpy as np
import poromics

im_3d = np.random.rand(50, 50, 50) > 0.4  # 3D porous image
result = poromics.permeability_lbm(im_3d, axis=0, nu=1e-6, voxel_size=1e-6)
print(result.k_m2, result.k_mD)
```

## Result objects

Both tortuosity functions return a `TortuosityResult`:

| Attribute          | Description                                    |
|--------------------|------------------------------------------------|
| `im`               | Boolean image used in the simulation           |
| `axis`             | Axis along which the simulation was run        |
| `porosity`         | Pore volume fraction                           |
| `tau`              | Tortuosity factor (≥ 1)                        |
| `D_eff`            | Normalized effective diffusivity (D_eff / D_0) |
| `c`                | Steady-state concentration field               |
| `formation_factor` | Formation factor F = 1 / D_eff                 |
| `D`                | Bulk diffusivity (float or ndarray)            |

`permeability_lbm` returns a `PermeabilityResult`:

| Attribute   | Description                            |
|-------------|----------------------------------------|
| `im`        | Boolean image used in the simulation   |
| `axis`      | Axis along which the simulation was run|
| `porosity`  | Pore volume fraction                   |
| `k_lu`      | Permeability in lattice units (voxels²)|
| `k_m2`      | Permeability in m²                     |
| `k_mD`      | Permeability in milliDarcy             |
| `u_darcy`   | Darcy (superficial) velocity in m/s    |
| `u_pore`    | Mean pore-space velocity in m/s        |
| `velocity`  | Steady-state velocity field (m/s)      |

## Simulation solvers

For more control over the simulation, use the solver classes directly. These accept physical SI units and expose intermediate state.

### Diffusion

```python
from poromics.simulation import TransientDiffusion

solver = TransientDiffusion(im, axis=1, D=1e-9, voxel_size=1e-6)
solver.run(n_steps=100_000, tol=1e-2)

c = solver.concentration    # Steady-state concentration field
J = solver.flux(axis=1)     # Mean flux at midplane
print(f"Converged: {solver.converged}, dt: {solver.dt:.2e} s")
```

### Flow

```python
from poromics.simulation import TransientFlow

solver = TransientFlow(im_3d, axis=0, nu=1e-6, voxel_size=1e-6)
solver.run(n_steps=100_000, tol=1e-3)

v = solver.velocity   # Velocity field in m/s, shape (nx, ny, nz, 3)
P = solver.pressure   # Pressure field in Pa (gauge)
print(f"Converged: {solver.converged}, dt: {solver.dt:.2e} s")
```

## Visualize the results

In addition to the tortuosity factor, `poromics` also provides the concentration field as a 2D/3D array, which can be useful, e.g., for training a machine learning model! You can visualize the concentration field using `matplotlib`.

```python
fig, ax = plt.subplots()
cax = ax.imshow(result.c[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Concentration Field")
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("c [mol/m^3]")
```

```python
# mkdocs: render
# mkdocs: hidecode
import poromics
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, seed=42)
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)

fig, ax = plt.subplots()
cax = ax.imshow(result.c[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Concentration Field")
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"c (mol/$m^3$)")
```
