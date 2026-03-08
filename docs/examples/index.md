# Examples

Worked examples demonstrating Poromics solvers on realistic porous media images. Each notebook walks through image generation, simulation, post-processing, and visualization.

| Example | Solver | Description |
|---------|--------|-------------|
| [Diffusion (FD)](diffusion-fd.ipynb) | `tortuosity_fd` | Tortuosity of a PoreSpy-generated porous medium using the Julia FD solver |
| [Diffusion (LBM)](diffusion-lbm.ipynb) | `tortuosity_lbm` | Tortuosity using the D3Q7 BGK lattice Boltzmann solver |
| [Flow (LBM)](flow-lbm.ipynb) | `permeability_lbm` | Permeability using the D3Q19 MRT lattice Boltzmann solver with streamline visualization |
