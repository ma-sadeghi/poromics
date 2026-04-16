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


def test_tau_lbm_sparse_matches_dense():
    """sparse=True must yield the same tau as sparse=False on a non-trivial geometry.

    Skips if the Taichi backend does not support pointer SNode (e.g. Metal).
    Probing sparse allocation directly would corrupt Taichi's global
    field registry on unsupported backends, so we query the arch name.
    """
    from poromics._lbm._taichi_helpers import ensure_taichi

    ti = ensure_taichi()
    try:
        arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1].lower()
    except Exception as e:
        pytest.skip(f"could not query Taichi arch ({e}); skipping sparse test")
    if arch not in ("cpu", "x64", "arm64", "cuda"):
        pytest.skip(f"sparse storage not supported on Taichi backend '{arch}'")

    im = np.zeros((20, 20, 20), dtype=bool)
    im[5:, :, :] = True  # half-open channel
    kwargs = dict(axis=0, voxel_size=1.0, n_steps=5000, tol=1e-3)
    r_dense = tortuosity_lbm(im, sparse=False, **kwargs)
    r_sparse = tortuosity_lbm(im, sparse=True, **kwargs)
    assert r_sparse.tau == pytest.approx(r_dense.tau, rel=1e-3)
