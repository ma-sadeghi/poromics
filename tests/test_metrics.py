import numpy as np
import porespy as ps
import pytest
from poromics._metrics import tortuosity_fd


def test_tau_open_space():
    """Tests the tortuosity calculation in open space."""
    # Create a 3D open space image
    im = np.ones((10, 10, 10), dtype=bool)
    # Calculate tortuosity along all principal axes
    result_x = tortuosity_fd(im, axis=0, rtol=1e-5, gpu=False)
    result_y = tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
    result_z = tortuosity_fd(im, axis=2, rtol=1e-5, gpu=False)
    # Check that the tortuosity is 1 for all axes
    assert np.allclose(result_x.tau, 1.0)
    assert np.allclose(result_y.tau, 1.0)
    assert np.allclose(result_z.tau, 1.0)


def test_tau_half_open_channel():
    """Tests the tortuosity calculation in a half-open channel."""
    # Create a 3D half-open channel image
    im = np.zeros((10, 10, 10), dtype=bool)
    im[5:, :, :] = 1
    # Calculate tortuosity along the orthogonal axes
    result_y = tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
    result_z = tortuosity_fd(im, axis=2, rtol=1e-5, gpu=False)
    # Channel is half-open, but eps is also 0.5, so tortuosity is still 1
    assert np.allclose(result_y.tau, 1.0)
    assert np.allclose(result_z.tau, 1.0)
    # Calculate tortuosity along the channel axis, which is blocked!
    with pytest.raises(RuntimeError):
        tortuosity_fd(im, axis=0, rtol=1e-5, gpu=False)


def test_tau_blobs():
    """Tests the tortuosity calculation in a blob-like structure."""
    # Create a 3D blob-like structure
    im = ps.generators.blobs(shape=[64, 64, 64], porosity=0.7)
    # Calculate tortuosity along all principal axes
    result_x = tortuosity_fd(im, axis=0, rtol=1e-5, gpu=False)
    result_y = tortuosity_fd(im, axis=1, rtol=1e-5, gpu=False)
    result_z = tortuosity_fd(im, axis=2, rtol=1e-5, gpu=False)
    # Check that the tortuosity is > 1 for all axes
    assert result_x.tau > 1.0
    assert result_y.tau > 1.0
    assert result_z.tau > 1.0


def test_tau_variable_diffusivity_full_domain():
    """Tests the tortuosity calculation with variable diffusivity in a full domain."""
    # Create a 3D image with variable diffusivity
    domain = np.ones((10, 10, 10), dtype=bool)
    im = ps.generators.blobs(shape=[10, 10, 10], porosity=0.7)
    # Create diffusivity field
    D = np.zeros_like(im, dtype=float)
    D[im] = 1.0  # Higher diffusivity in the blob regions
    D[~im] = 0.5  # Lower diffusivity in the non-blob regions
    # Calculate tortuosity along the channel axis
    result = tortuosity_fd(domain, D=D, axis=0, rtol=1e-5, gpu=False)
    # Check that the tortuosity is > 1 due to variable diffusivity
    assert result.tau > 1.0


def test_tau_variable_diffusivity_subdomain():
    """Tests the tortuosity calculation with variable diffusivity in a subdomain."""
    # Create a 3D image with variable diffusivity
    im = ps.generators.blobs(shape=[10, 10, 10], porosity=0.7)
    # Create random diffusivity field
    D = np.random.rand(*im.shape)
    D[~im] = 0.0  # NO diffusivity in the non-blob regions
    # Only consider regions with non-zero diffusivity
    domain = D > 0
    # Calculate tortuosity along the channel axis
    result = tortuosity_fd(domain, D=D, axis=0, rtol=1e-5, gpu=False)
    # Check that the tortuosity is > 1 due to variable diffusivity
    assert result.tau > 1.0
