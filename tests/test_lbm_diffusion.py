# Tests for LBM-based diffusion/tortuosity solver.
import numpy as np
import pytest
from poromics._metrics import tortuosity_lbm


def test_tau_lbm_open_space():
    """Open space should have tortuosity close to 1."""
    im = np.ones((20, 20, 20), dtype=bool)
    result = tortuosity_lbm(im, axis=0, voxel_size=1.0, n_steps=5000, tol=1e-2)
    assert np.isclose(result.tau, 1.0, atol=0.1)
    assert np.isclose(result.D_eff, result.porosity, atol=0.1)


def test_tau_lbm_half_open():
    """Half-open channel: tau should still be ~1 along open axes."""
    im = np.zeros((20, 20, 20), dtype=bool)
    im[10:, :, :] = True
    result = tortuosity_lbm(im, axis=1, voxel_size=1.0, n_steps=5000, tol=1e-2)
    assert np.isclose(result.tau, 1.0, atol=0.15)


def test_tau_lbm_no_pores():
    """Fully solid image should raise RuntimeError."""
    im = np.zeros((10, 10, 10), dtype=bool)
    with pytest.raises(RuntimeError):
        tortuosity_lbm(im, axis=0, voxel_size=1.0)


def test_tau_lbm_invalid_axis():
    """Invalid axis should raise ValueError."""
    im = np.ones((10, 10, 10), dtype=bool)
    with pytest.raises(ValueError):
        tortuosity_lbm(im, axis=3, voxel_size=1.0)


def test_tau_lbm_converged_true_on_easy_case():
    im = np.ones((10, 10, 10), dtype=bool)
    result = tortuosity_lbm(im, axis=0, voxel_size=1.0, n_steps=10000, tol=1e-2)
    assert result.converged is True
    assert result.n_iterations is not None
    assert result.n_iterations > 0


def test_tau_lbm_converged_false_when_steps_exhausted():
    from loguru import logger as _loguru

    messages = []
    handler_id = _loguru.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        im = np.ones((10, 10, 10), dtype=bool)
        result = tortuosity_lbm(im, axis=0, voxel_size=1.0, n_steps=1, tol=1e-12)
    finally:
        _loguru.remove(handler_id)
    assert result.converged is False
    assert result.n_iterations > 0
    assert any("did not converge" in m for m in messages)
