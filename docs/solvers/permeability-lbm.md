# Permeability (LBM solver)

## Background

This solver computes absolute permeability by solving the incompressible Navier-Stokes equations in the Stokes (creeping flow) regime using LBM:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}$$

At low Reynolds numbers the inertial term vanishes, leaving Stokes flow. The solver uses the D3Q19 lattice (19 velocities in 3D) with the MRT (multiple-relaxation-time) collision operator, which improves numerical stability compared to BGK by relaxing different moments at independent rates.

Boundary conditions are fixed-pressure (density) on the inlet and outlet faces, with periodic conditions on the remaining faces. Solid voxels use bounce-back. Once the velocity field reaches steady state, permeability is extracted via Darcy's law:

$$k = \frac{\mu \, u_\text{Darcy}}{\Delta p / L}$$

where $u_\text{Darcy}$ is the volume-averaged (superficial) velocity, $\Delta p / L$ is the pressure gradient, and $\mu = \rho \nu$ is the dynamic viscosity.

The solver accepts physical SI units for viscosity and voxel size; lattice parameters are computed internally.

## Usage

```python
# mkdocs: render
# mkdocs: hidecode
# mkdocs: hideoutput
import poromics
import porespy as ps
import matplotlib.pyplot as plt
import numpy as np

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, blobiness=0.5, seed=42)
result = poromics.permeability_lbm(im, axis=1, nu=1e-6, voxel_size=1e-6)
```

```python exec="1" session="perm"
# mkdocs: hidecode
import poromics
import porespy as ps

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, blobiness=0.5, seed=42)
result = poromics.permeability_lbm(im, axis=1, nu=1e-6, voxel_size=1e-6)
```

```python exec="1" session="perm"
print(f"Permeability: {result.k_m2:.4e} m² ({result.k_mD:.2f} mD)")
```

## Result

`permeability_lbm` returns a `PermeabilityResult`:

| Attribute   | Description                                |
|-------------|--------------------------------------------|
| `im`        | Boolean image used in the simulation       |
| `axis`      | Axis along which the simulation was run    |
| `porosity`  | Pore volume fraction                       |
| `k_lu`      | Permeability in lattice units (voxels$^2$) |
| `k_m2`      | Permeability in m$^2$                      |
| `k_mD`      | Permeability in milliDarcy                 |
| `u_darcy`   | Darcy (superficial) velocity in m/s        |
| `u_pore`    | Mean pore-space velocity in m/s            |
| `velocity`  | Steady-state velocity field (m/s)          |

## Visualize the velocity field

The velocity field can be visualized as a streamline plot. For a quasi-2D image, we extract the `z=0` slice.

```python
# mkdocs: render
# mkdocs: hidecode
v = result.velocity[:, :, 0, :]
speed = np.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)

fig, ax = plt.subplots()
ax.imshow(im[:, :, 0], cmap="gray", interpolation="nearest", alpha=0.3)
nx, ny = im.shape[0], im.shape[1]
X, Y = np.meshgrid(np.arange(ny), np.arange(nx))
lw = 3 * speed / speed.max()
ax.streamplot(X, Y, v[:, :, 1], v[:, :, 0], color=speed,
              cmap="viridis", density=2, linewidth=lw)
ax.set_title("Velocity Streamlines")
```
