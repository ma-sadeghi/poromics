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
print(f"Permeability: {result.k:.4e} m²")
```

## Result

`permeability_lbm` returns a `PermeabilityResult`:

| Attribute   | Description                                |
|-------------|--------------------------------------------|
| `im`        | Boolean image used in the simulation       |
| `axis`      | Axis along which the simulation was run    |
| `porosity`  | Pore volume fraction                       |
| `k`         | Permeability in m$^2$                      |
| `u_darcy`   | Darcy (superficial) velocity in m/s        |
| `u_pore`    | Mean pore-space velocity in m/s            |
| `velocity`  | Steady-state velocity field (m/s)          |
| `pressure`  | Gauge pressure field (Pa)                  |

## Rescaling to different physical parameters

Because permeability depends only on geometry and the Stokes equations are linear, the simulation results can be reinterpreted for a different voxel size, fluid viscosity, or fluid density without rerunning the solver:

```python exec="1" session="perm"
rescaled = result.rescale(voxel_size=5e-6, nu=0.5e-6, rho=800)
print(f"Rescaled k: {rescaled.k:.4e} m²")
print(f"Rescaled u_darcy: {rescaled.u_darcy:.4e} m/s")
```

The `rho` parameter is optional. When provided, the pressure field is converted to Pa; when omitted, `pressure` is set to `None`.

!!! note
    `TortuosityResult` does not have a `rescale` method because tortuosity, effective diffusivity, and the concentration field are all dimensionless.

## Visualize the velocity field

The velocity field can be visualized as a streamline plot. For a quasi-2D image, we extract the `z=0` slice.

```python
# mkdocs: render
# mkdocs: hidecode
result.plot_velocity(z=0)
```
