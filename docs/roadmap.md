# Roadmap

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
