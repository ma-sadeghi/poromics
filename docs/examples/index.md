# Examples

Worked examples demonstrating Poromics solvers on realistic porous media images. Each notebook walks through image generation, simulation, post-processing, and visualization.

| Example | Solver | Description |
|---------|--------|-------------|
| [Tortuosity (FDM)](diffusion-fd.ipynb) | `tortuosity_fd` | Tortuosity via FDM diffusion solver |
| [Tortuosity (LBM)](diffusion-lbm.ipynb) | `tortuosity_lbm` | Tortuosity via D3Q7 BGK lattice Boltzmann solver |
| [Permeability (LBM)](flow-lbm.ipynb) | `permeability_lbm` | Permeability via D3Q19 MRT lattice Boltzmann solver |
| [Rescaling Results](rescaling.ipynb) | `PermeabilityResult.rescale` | Reinterpret flow results for different fluids or voxel sizes |
