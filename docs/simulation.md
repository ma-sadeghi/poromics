# Simulation API

For more control over the simulation, use the solver classes directly. These accept physical SI units and expose intermediate state for custom post-processing.

The high-level functions (`tortuosity_lbm`, `permeability_lbm`) use these classes internally.

## TransientDiffusion

Solves the transient diffusion equation using LBM (D3Q7 BGK).

```python
from poromics.simulation import TransientDiffusion

solver = TransientDiffusion(im, axis=1, D=1e-9, voxel_size=1e-6)
solver.run(n_steps=100_000, tol=1e-2)

c = solver.concentration    # Steady-state concentration field
J = solver.flux(axis=1)     # Mean flux at midplane
print(f"Converged: {solver.converged}, dt: {solver.dt:.2e} s")
```

| Property / Method   | Description                                      |
|----------------------|--------------------------------------------------|
| `concentration`      | Concentration field, shape (nx, ny, nz)          |
| `flux(axis)`         | Mean diffusive flux at the domain midplane        |
| `converged`          | Whether convergence was reached                   |
| `n_iterations`       | Total number of time steps taken                  |
| `dt`                 | Physical time step in seconds                     |
| `voxel_size`         | Voxel edge length in metres                       |
| `step()`             | Advance by one time step                          |
| `run(n_steps, tol)`  | Run to steady state                               |

## TransientFlow

Solves incompressible Stokes flow using LBM (D3Q19 MRT).

```python
from poromics.simulation import TransientFlow

solver = TransientFlow(im, axis=1, nu=1e-6, voxel_size=1e-6)
solver.run(n_steps=100_000, tol=1e-3)

v = solver.velocity   # Velocity field in m/s, shape (nx, ny, nz, 3)
P = solver.pressure   # Pressure field in Pa (gauge)
print(f"Converged: {solver.converged}, dt: {solver.dt:.2e} s")
```

| Property / Method    | Description                                      |
|----------------------|--------------------------------------------------|
| `velocity`           | Velocity field in m/s, shape (nx, ny, nz, 3)    |
| `pressure`           | Gauge pressure field in Pa, shape (nx, ny, nz)   |
| `converged`          | Whether convergence was reached                   |
| `n_iterations`       | Total number of time steps taken                  |
| `dt`                 | Physical time step in seconds                     |
| `voxel_size`         | Voxel edge length in metres                       |
| `step()`             | Advance by one time step                          |
| `run(n_steps, tol)`  | Run to steady state                               |
