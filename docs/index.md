# Poromics documentation

## What is Poromics?

![Poromics logo](assets/logo-old.png){ width="320px" align=right }

Poromics is an open-source Python package for estimation of transport properties of 3D images of porous materials. The main design philosophies driving Poromics are speed and ease of use. Poromics is GPU-accelerated via [Taichi](https://www.taichi-lang.org/) (LBM solvers) and optionally via Julia/CUDA ([Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/) FD solver).

**Supported properties:**

- **Tortuosity / effective diffusivity** — via Julia FD solver (`tortuosity_fd`) or Taichi LBM D3Q7 BGK (`tortuosity_lbm`)
- **Absolute permeability** — via Taichi LBM D3Q19 MRT (`permeability_lbm`)

![Tortuosity simulation workflow](assets/workflow.png){ width="100%" }
/// caption
A typical workflow for estimating tortuosity. The user loads a 2D/3D image of a porous material and specifies an axis. Poromics then performs a steady-state diffusion simulation along the specified axis to compute the tortuosity from the concentration profile [^1].
///

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

## Acknowledgments

The LBM solvers are based on [taichi_LBM3D](https://github.com/yjhp1016/taichi_LBM3D) by Yi-Jie Huang. The FD tortuosity solver uses [Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/).

## Contributing

We welcome contributions to Poromics! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. For more information on how to contribute, please refer to the [contributing guide](contributing.md), or simply open an [issue](https://github.com/ma-sadeghi/poromics/issues) or a pull [request](https://github.com/ma-sadeghi/poromics/pulls) on GitHub.

[^1]: Image from: Fu, J., Thomas, H. R., & Li, C. (2021). [Tortuosity of porous media: Image analysis and physical simulation](https://doi.org/10.1016/j.earscirev.2020.103439). *Earth-Science Reviews*, 212, 103439. (Figure 9)
