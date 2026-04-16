# Tests for the persistent Julia subprocess worker lifecycle.
import numpy as np
import pytest

from poromics import _metrics
from poromics._metrics import tortuosity_fd


@pytest.fixture(autouse=True)
def _reset_worker():
    """Force a fresh worker for each test; clean up on exit."""
    _metrics._shutdown_julia_worker()
    yield
    _metrics._shutdown_julia_worker()


def _small_open_image():
    return np.ones((6, 6, 6), dtype=bool)


def test_worker_is_spawned_on_first_call():
    assert _metrics._julia_proc is None
    tortuosity_fd(_small_open_image(), axis=0, rtol=1e-5, gpu=False)
    assert _metrics._julia_proc is not None
    assert _metrics._julia_proc.poll() is None  # still alive


def test_worker_is_reused_across_calls():
    """Second call must hit the same Julia subprocess (JIT cache warm)."""
    tortuosity_fd(_small_open_image(), axis=0, rtol=1e-5, gpu=False)
    pid_first = _metrics._julia_proc.pid
    tortuosity_fd(_small_open_image(), axis=1, rtol=1e-5, gpu=False)
    pid_second = _metrics._julia_proc.pid
    assert pid_first == pid_second


def test_worker_error_propagates_as_python_exception():
    """A Julia-side failure must come back as a Python exception."""
    # An all-solid image trips the "no percolating paths" check in
    # the public wrapper before reaching Julia, so instead we force
    # Julia itself to fail by passing an axis the Julia side rejects.
    im = _small_open_image()
    # Monkey-patch the payload dict inspector to confirm the worker
    # survives a crash: directly call _julia_call with a bad payload.
    with pytest.raises((RuntimeError, Exception)):
        _metrics._julia_call(
            {
                "im": im,
                "axis": 99,  # invalid -- Julia indexes ["x","y","z"][99]
                "D": None,
                "rtol": 1e-5,
                "gpu": False,
                "verbose": False,
            }
        )
    # Worker should still be alive and usable after the failure.
    assert _metrics._julia_proc.poll() is None
    result = tortuosity_fd(im, axis=0, rtol=1e-5, gpu=False)
    assert np.isclose(result.tau, 1.0)


def test_worker_shutdown_terminates_process():
    tortuosity_fd(_small_open_image(), axis=0, rtol=1e-5, gpu=False)
    proc = _metrics._julia_proc
    assert proc.poll() is None
    _metrics._shutdown_julia_worker()
    # After shutdown the module-level handles are cleared and the
    # process has exited.
    assert _metrics._julia_proc is None
    assert proc.poll() is not None
