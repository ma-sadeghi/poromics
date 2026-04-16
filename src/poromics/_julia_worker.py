# Persistent subprocess worker for Julia-based solvers.
#
# Runs a request loop so Julia + JIT stay warm across calls. Avoids LLVM
# symbol collisions between Julia and Taichi by keeping them in separate
# processes. Communication uses pickle tempfiles; file paths are exchanged
# over a dedicated pipe fd (not stdin, which Julia's runtime clobbers).

import os
import pickle
import sys

os.environ["PYTHON_JULIACALL_STARTUP_FILE"] = "no"
os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"

_jl = None
_taujl = None


def _ensure_julia():
    """Initialize Julia and load Tortuosity.jl (once per process)."""
    global _jl, _taujl
    if _jl is None:
        from poromics import julia_helpers

        julia_helpers.ensure_julia_deps_ready(quiet=False)
        _jl = julia_helpers.init_julia(quiet=False)
        _taujl = julia_helpers.import_backend(_jl)
    return _jl, _taujl


def _run_tortuosity_fd(im, axis, D, rtol, gpu, verbose):
    """Execute the Julia-backed tortuosity solver and return a plain dict."""
    import numpy as np

    jl, taujl = _ensure_julia()

    axis_jl = jl.Symbol(["x", "y", "z"][axis])
    sim = taujl.TortuositySimulation(im, D=D, axis=axis_jl, gpu=gpu)
    sol = taujl.solve(sim.prob, taujl.KrylovJL_CG(), verbose=verbose, reltol=rtol)
    c = taujl.vec_to_grid(sol.u, im)
    tau = taujl.tortuosity(c, axis=axis_jl, D=D)
    D_eff = taujl.effective_diffusivity(c, axis=axis_jl, D=D)

    pore_mask = np.asarray(im, dtype=bool)
    porosity = float(pore_mask.sum()) / pore_mask.size
    formation_factor = 1.0 / D_eff if D_eff > 0 else float("inf")

    return {
        "im": np.asarray(im, dtype=bool),
        "axis": axis,
        "porosity": porosity,
        "tau": float(tau),
        "D_eff": float(D_eff),
        "c": np.asarray(c),
        "formation_factor": float(formation_factor),
        "D": D,
        "converged": True,
        "n_iterations": None,
    }


def main():
    """Run a persistent request loop, reading file paths from a pipe fd.

    Redirects fd 1 (stdout) to /dev/null at startup so Julia's verbose
    output never pollutes the protocol channel. A dedicated file object
    wrapping the original stdout pipe fd is used for "done" signals.
    """
    ctrl_fd = int(sys.argv[1])
    ctrl_in = os.fdopen(ctrl_fd, "r")

    # Save the real stdout pipe, then redirect fd 1 to /dev/null so
    # Julia/juliacall output can never reach the parent's readline().
    signal_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    signal_out = os.fdopen(signal_fd, "w")

    while True:
        line = ctrl_in.readline()
        if not line:
            break
        in_path, out_path = line.strip().split("\t")
        with open(in_path, "rb") as f:
            payload = pickle.load(f)
        try:
            result = _run_tortuosity_fd(**payload)
            response = {"ok": True, "result": result}
        except Exception as e:
            response = {
                "ok": False,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            }
        with open(out_path, "wb") as f:
            pickle.dump(response, f)
        # Signal completion — parent reads this line to know the result is ready
        signal_out.write("done\n")
        signal_out.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
