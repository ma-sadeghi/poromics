# Basic usage

After installing `poromics`, you can use it in your Python scripts or Jupyter notebooks. The package provides a simple interface to calculate tortuosity using the `Tortuosity.jl` package. Here's a basic example of how to use it:

!!! note

    The first time you import `poromics`, it will take a few minutes to install Julia and the required packages. This is a one-time setup.

## Generate a test image

You can use the `porespy` package to generate a test image. In this example, we will create a simple 2D image with `blobs`.

```python
# mkdocs: render
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, seed=42)

fig, ax = plt.subplots()
ax.imshow(im[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Boolean Image")
```

## Perform tortuosity simulation

```python
import poromics

result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=True)
print(result)
```

```python exec="1"
# mkdocs: hidecode
import poromics
import porespy as ps

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, seed=42)
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
print(result)
```

The `Result` object is a simple container with the following attributes:

- `im`: The tortuosity factor of the image.
- `axis`: The axis along which the tortuosity was calculated.
- `tau`: The tortuosity factor.
- `c`: The concentration field.

## Visualize the results

In addition to the tortuosity factor, `poromics` also provides the concentration field as a 2D/3D array, which can be useful, e.g., for training a machine learning model! You can visualize the concentration field using `matplotlib`.

```python
fig, ax = plt.subplots()
cax = ax.imshow(result.c[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Concentration Field")
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("c [mol/m^3]")
```

```python
# mkdocs: render
# mkdocs: hidecode
import poromics
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, seed=42)
result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)

fig, ax = plt.subplots()
cax = ax.imshow(result.c[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Concentration Field")
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"c (mol/$m^3$)")
```

## Full example

```python
import poromics
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[100, 100, 1], porosity=0.6, seed=42)

fig, ax = plt.subplots()
ax.imshow(im[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Boolean Image")

result = poromics.tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
print(result)

fig, ax = plt.subplots()
cax = ax.imshow(result.c[:, :, 0], cmap="viridis", interpolation="nearest")
ax.set_title("Concentration Field")
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("c [mol/m^3]")
```
