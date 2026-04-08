# Poromics

Poromics estimates transport properties of 3D porous material images. It is GPU-accelerated and designed to be fast and easy to use.

**Supported properties:**

- **Tortuosity / effective diffusivity** — via Julia-based FD solver (`tortuosity_fd`) or Taichi-based LBM D3Q7 BGK solver (`tortuosity_lbm`)
- **Absolute permeability** — via Taichi-based LBM D3Q19 MRT solver (`permeability_lbm`)

## Installation

The Julia-based FD solver depends on [Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/), which is installed automatically. The LBM solvers use [Taichi](https://www.taichi-lang.org/) with automatic GPU detection.

> [!NOTE]
> We highly recommend using `uv` instead of `pip` to install `poromics` (or any other Python package!) as it's extremely faster. It has lots of useful features, but for all practical purposes, it is a drop-in replacement for `pip`.

### Uv

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), and then run the following command in a terminal/command prompt:

```shell
uv pip install poromics
```

### Pip

If you prefer to use `pip`, run the following command in a terminal/command prompt:

```shell
pip install poromics
```

## Basic Usage

> [!NOTE]
> The first time you call `tortuosity_fd`, it will take a few minutes to install Julia and the required packages. This is a one-time setup. The LBM solvers (`tortuosity_lbm`, `permeability_lbm`) use Taichi and do not require Julia.

### Tortuosity (Julia FD solver)

```python
import porespy as ps
import poromics

im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.6)
result = poromics.tortuosity_fd(im, axis=0, rtol=1e-5, gpu=True)
print(result.tau, result.D_eff)
```

### Tortuosity (LBM solver)

```python
result = poromics.tortuosity_lbm(im, axis=0, D=1e-9, voxel_size=1e-6)
print(result.tau, result.D_eff)
```

### Permeability (LBM solver)

```python
result = poromics.permeability_lbm(im, axis=0, nu=1e-6, voxel_size=1e-6)
print(result.k)
```

### Result objects

`TortuosityResult` attributes: `im`, `axis`, `porosity`, `tau`, `D_eff`, `c`, `formation_factor`, `D`.

`PermeabilityResult` attributes: `im`, `axis`, `porosity`, `k`, `u_darcy`, `u_pore`, `velocity`, `pressure`.

### Simulation solvers

For more control, use the solver classes directly:

```python
from poromics.simulation import TransientDiffusion, TransientFlow

solver = TransientDiffusion(im, axis=0, D=1e-9, voxel_size=1e-6)
solver.run(n_steps=100_000, tol=1e-2)
print(solver.concentration.shape, solver.converged)
```

## CLI

> [!WARNING]
> The CLI is still in development and not yet functional.

```bash
poromics --help
```

## Acknowledgments

The LBM solvers are based on [taichi_LBM3D](https://github.com/yjhp1016/taichi_LBM3D) by Yi-Jie Huang.

## Roadmap

- [x] Diffusional tortuosity
    - [x] Julia FD solver ([Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/))
    - [x] Taichi LBM D3Q7 BGK solver
- [ ] Transient tortuosity
    - [ ] Julia FD solver ([Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/))
- [x] Permeability
    - [x] Taichi LBM D3Q19 MRT solver
- [ ] [Electrode tortuosity](https://doi.org/10.1038/s41524-020-00386-4)
- [x] Julia/Taichi coexistence via subprocess isolation
- [ ] Add command-line interface (CLI) for easy usage
- [ ] Add support for [sysimage](https://julialang.github.io/PackageCompiler.jl/dev/sysimages.html) creation upon installation for faster startup
