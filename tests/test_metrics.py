import subprocess
import sys
import textwrap

import numpy as np
import porespy as ps
import pytest
from poromics._metrics import tortuosity_fd
from poromics import julia_helpers


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


def test_gpu_unsupported_backend_falls_back_to_cpu(monkeypatch):
    """With an unrecognized backend, tortuosity_fd should warn and run on CPU."""
    from poromics import _metrics

    monkeypatch.setenv("POROMICS_GPU_BACKEND", "bogus")
    # Restart the Julia worker so the patched env is inherited by the subprocess.
    _metrics._shutdown_julia_worker()
    im = np.ones((6, 6, 6), dtype=bool)
    result = tortuosity_fd(im, axis=0, rtol=1e-5)  # default gpu=True
    assert np.isclose(result.tau, 1.0)


def test_detect_gpu_backend_env_override(monkeypatch):
    monkeypatch.setenv("POROMICS_GPU_BACKEND", "metal")
    assert julia_helpers._detect_gpu_backend() == "Metal"
    monkeypatch.setenv("POROMICS_GPU_BACKEND", "cuda")
    assert julia_helpers._detect_gpu_backend() == "CUDA"
    monkeypatch.setenv("POROMICS_GPU_BACKEND", "amdgpu")
    assert julia_helpers._detect_gpu_backend() == "AMDGPU"


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


def test_tau_open_space_gpu():
    """τ must equal 1.0 on open space even on the GPU path.

    Regression against the Tortuosity.jl v0.0.6 atomic-kernel bug that made
    Metal return τ ≈ 0.3-0.9 on open cubes. Falls back to CPU silently on
    platforms without a functional GPU backend; the assertion still catches
    numerical regressions on the CPU path.
    """
    im = np.ones((10, 10, 10), dtype=bool)
    for axis in (0, 1, 2):
        result = tortuosity_fd(im, axis=axis, rtol=1e-5, gpu=True, verbose=False)
        assert np.isclose(result.tau, 1.0), f"axis={axis}: tau={result.tau}"


def test_tortuosity_fd_gpu_no_sigbus_across_fresh_processes(tmp_path):
    """Regression test for juliacall --handle-signals=no SIGBUS.

    Before setting PYTHON_JULIACALL_HANDLE_SIGNALS=yes, fresh Python
    processes aborted ~60% of the time during Metal initialization on
    Darwin/arm64. Spawns several fresh subprocesses and asserts every one
    exits cleanly. Skipped when no GPU backend is configured — the test
    is meaningful only on a machine where Metal/CUDA/AMDGPU is actually
    exercised.
    """
    if julia_helpers._detect_gpu_backend() is None:
        pytest.skip("no GPU backend for this platform/override")
    script = tmp_path / "gpu_probe.py"
    script.write_text(textwrap.dedent("""
        import numpy as np
        from poromics import tortuosity_fd
        im = np.ones((6, 6, 6), dtype=bool)
        r = tortuosity_fd(im, axis=0, rtol=1e-5, gpu=True, verbose=False)
        assert 0.99 < r.tau < 1.01, f"tau out of range: {r.tau}"
    """))
    n_runs = 5
    failures = []
    for i in range(n_runs):
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, timeout=300
        )
        if proc.returncode != 0:
            failures.append(
                f"run {i}: exit={proc.returncode}\n"
                f"stderr={proc.stderr.decode(errors='replace')[-500:]}"
            )
    assert (
        not failures
    ), f"{len(failures)}/{n_runs} fresh-process runs failed:\n" + "\n---\n".join(
        failures
    )
