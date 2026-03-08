# Solvers

Poromics provides three solvers for estimating transport properties of porous media. Each solver page below describes the governing physics, boundary conditions, usage, and output fields.

| Solver | Property | Method | Backend |
|--------|----------|--------|---------|
| [`tortuosity_fd`](tortuosity-fd.md) | Tortuosity / $D_\text{eff}$ | Finite differences (Krylov) | Julia |
| [`tortuosity_lbm`](tortuosity-lbm.md) | Tortuosity / $D_\text{eff}$ | Lattice Boltzmann (D3Q7 BGK) | Taichi |
| [`permeability_lbm`](permeability-lbm.md) | Permeability | Lattice Boltzmann (D3Q19 MRT) | Taichi |

## Choosing a solver

**For tortuosity**, both solvers compute the same quantity — the ratio of the bulk diffusivity to the effective diffusivity through the pore space. The FD solver is generally faster for a single evaluation and supports spatially variable diffusivity fields. The LBM solver accepts physical units directly and provides the transient concentration field.

**For permeability**, only the LBM solver is available. It solves creeping (Stokes) flow and extracts permeability via Darcy's law.

All solvers automatically trim non-percolating pore regions before running, so dead-end pores that don't connect inlet to outlet are excluded.
