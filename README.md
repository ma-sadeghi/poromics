# Poromics

Poromics is a set of tools for rapid estimation of transport properties of 3D images of porous materials. It is designed to be fast and easy to use. Currently, it can predict the tortuosity factor of an image. The goal is to support more transport properties in the future such as permeability. Poromics is optionally GPU-accelerated, which can significantly speed up the calculations for large images (up to 100x speedup).

## Installation

Poromics depends on the Julia package [Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/). However, it is not necessary to install Julia separately. The package will be installed automatically when you install `poromics`.

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
> The first time you import `poromics`, it will take a few minutes to install Julia and the required packages. This is a one-time setup.

```python
import porespy as ps
import poromics

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6)  # Test image
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=True)
print(result)
```

The `Result` object is a simple container with the following attributes:

- `im`: The tortuosity factor of the image.
- `axis`: The axis along which the tortuosity was calculated.
- `tau`: The tortuosity factor.
- `c`: The concentration field.

## CLI

> [!WARNING]  
> The CLI is still in development and not yet functional.

```bash
poromics --help
```

## Roadmap

- [ ] Speed up matrix assembly by direct assembly on GPU.
- [ ] Create Julia sysimage files upon installation for faster startup.
- [ ] Add more transport properties (e.g. permeability).
- [ ] Add CLI support.
