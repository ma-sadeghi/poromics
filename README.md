# Poromics

Poromics is a set of tools for rapid estimation of transport properties of 3D images of porous materials. It is designed to be fast and easy to use. Currently, it can predict the tortuosity factor of an image. The goal is to support more transport properties in the future such as permeability. Poromics is optionally GPU-accelerated, which can significantly speed up the calculations for large images (up to 100x speedup).

## Installation

Poromics depends on the Julia package [Tortuosity.jl](https://github.com/ma-sadeghi/Tortuosity.jl/). However, it is not necessary to install Julia separately. The package will be installed automatically when you install Poromics.

```bash
pip install poromics
```

## Basic Usage

```python
import porespy as ps
import poromics

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6)  # Test image
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=True)
print(result)  # result has the following attributes: im, axis, tau, c
```

## CLI

> [!WARNING]  
> The CLI is still in development and not yet functional.

```bash
poromics --help
```
