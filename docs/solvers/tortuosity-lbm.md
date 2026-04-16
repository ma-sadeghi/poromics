# Tortuosity (LBM solver)

## Background

This solver computes tortuosity by solving the transient diffusion equation using the Lattice Boltzmann Method (LBM):

$$\frac{\partial c}{\partial t} = \nabla \cdot (D \, \nabla c)$$

The solver uses the D3Q7 lattice (7 velocities in 3D) with the BGK (single-relaxation-time) collision operator. Each lattice node carries 7 distribution functions $f_i$ that evolve via the collide-and-stream algorithm:

$$f_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) = f_i(\mathbf{x}, t) - \frac{1}{\tau_\text{BGK}} \left[ f_i - f_i^\text{eq} \right]$$

where $\tau_\text{BGK}$ is the relaxation time, related to the diffusivity by $D = c_s^2 (\tau_\text{BGK} - 0.5) \Delta t$.

Boundary conditions are Dirichlet (fixed concentration) on inlet/outlet faces and periodic on the remaining faces. Solid voxels use bounce-back. The simulation runs until the relative change in the concentration field drops below the tolerance, then tortuosity is extracted from the steady-state flux via Fick's law.

The solver accepts physical SI units for diffusivity and voxel size; lattice parameters are computed internally.

## Usage

```python exec="1" session="tau-lbm"
# mkdocs: hidecode
import poromics
import porespy as ps

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, blobiness=0.5, seed=42)
result = poromics.tortuosity_lbm(im, axis=1, D=1e-9, voxel_size=1e-6)
```

```python exec="1" session="tau-lbm"
print(f"Tortuosity: {result.tau:.4f}")
print(f"Effective diffusivity (D_eff/D_0): {result.D_eff:.6f}")
print(f"Formation factor: {result.formation_factor:.4f}")
```

## Result

`tortuosity_lbm` returns a `TortuosityResult`:

| Attribute          | Description                                    |
|--------------------|------------------------------------------------|
| `im`               | Boolean image used in the simulation           |
| `axis`             | Axis along which the simulation was run        |
| `porosity`         | Pore volume fraction                           |
| `tau`              | Tortuosity factor ($\geq 1$)                   |
| `D_eff`            | Normalized effective diffusivity ($D_\text{eff} / D_0$) |
| `c`                | Steady-state concentration field               |
| `formation_factor` | Formation factor $F = 1 / D_\text{eff}$        |
| `D`                | Bulk diffusivity                               |
| `converged`        | Whether the solver reached the requested tolerance |
| `n_iterations`     | Number of LBM iterations taken (None for the FD solver) |

!!! warning "Check `converged` before trusting `tau`"
    If the maximum step count is reached without convergence,
    `tortuosity_lbm` emits a warning and sets `result.converged =
    False`; the reported `tau` is then from a pre-steady-state field.
    Always check this flag on slow-to-converge geometries, and
    consider raising `n_steps` or loosening `tol`.

### Tolerance and step-count guidance

`tol` measures the relative change in the concentration field between
samples taken 500 steps apart — it is a *rate* of change, not an
error bound. Tortuosity is noticeably more sensitive to tolerance
than permeability because it is derived from a gradient, not a
volume average.

Empirically, on 40³ blob images:

| Porosity | `tol=1e-2`  | `tol=1e-3` (default) | `tol=1e-4` |
|---------:|------------:|---------------------:|-----------:|
| 0.7      | ~1,500 (0.3 %) | ~3,000 (0.03 %)   | ~5,500     |
| 0.5      | ~2,000 (0.8 %) | ~5,000 (0.04 %)   | ~9,000     |
| 0.3      | ~4,000 (0.7 %) | ~9,500 (0.08 %)   | ~21,000    |

(Numbers in parentheses are the relative error in `tau` compared to
the tighter `tol=1e-4` reference.) At tight porosities `tol=1e-2` can
leave tau off by up to ~1 %, which is why the default is `1e-3`.

Diffusive relaxation time scales like $L^2 / D$, so iteration counts
scale roughly as $(L/40)^2$ for larger domains.

!!! note
    Unlike `PermeabilityResult`, `TortuosityResult` does not have a `rescale` method. Tortuosity, effective diffusivity, and the concentration field are all dimensionless quantities that do not depend on physical units.
