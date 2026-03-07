# Tests for LBM-based permeability solver.
import numpy as np
import pytest
from poromics._permeability import permeability_lbm


def test_k_lbm_open_space():
    """Open space should have positive, finite permeability."""
    im = np.ones((20, 20, 20), dtype=bool)
    result = permeability_lbm(im, axis=0, n_steps=5000, tol=1e-3)
    assert result.k_lu > 0
    assert np.isfinite(result.k_lu)
    assert result.porosity == 1.0


def test_k_lbm_no_pores():
    """Fully solid image should raise RuntimeError."""
    im = np.zeros((10, 10, 10), dtype=bool)
    with pytest.raises(RuntimeError):
        permeability_lbm(im, axis=0)


def test_k_lbm_invalid_axis():
    """Invalid axis should raise ValueError."""
    im = np.ones((10, 10, 10), dtype=bool)
    with pytest.raises(ValueError):
        permeability_lbm(im, axis=3)


def test_k_lbm_physical_units():
    """Physical units should be computed when dx_m is provided."""
    im = np.ones((20, 20, 20), dtype=bool)
    result = permeability_lbm(
        im,
        axis=0,
        n_steps=5000,
        tol=1e-3,
        dx_m=1e-6,
    )
    assert result.k_m2 is not None
    assert result.k_mD is not None
    assert result.k_m2 > 0
