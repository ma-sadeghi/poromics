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
    """Initialize Julia and load Tortuosity.jl (once per process).

    Redirects stdout to /dev/null during initialization so Julia's
    verbose output doesn't pollute the stdout protocol channel used
    to signal completion to the parent process.
    """
    global _jl, _taujl
    if _jl is None:
        saved_stdout_fd = os.dup(1)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.close(devnull_fd)
        try:
            from poromics import julia_helpers

            julia_helpers.ensure_julia_deps_ready(quiet=False)
            _jl = julia_helpers.init_julia(quiet=False)
            _taujl = julia_helpers.import_backend(_jl)
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
    return _jl, _taujl


def _run_tortuosity_fd(im, axis, D, rtol, gpu, verbose):
    """Execute the Julia-backed tortuosity solver and return a plain dict."""
    import numpy as np

    jl, taujl = _ensure_julia()

    axis_jl = jl.Symbol(["x", "y", "z"][axis])
    eps0 = taujl.Imaginator.phase_fraction(im)
    im = np.array(taujl.Imaginator.trim_nonpercolating_paths(im, axis=axis_jl))
    if jl.sum(im) == 0:
        raise RuntimeError("No percolating paths along the given axis found in the image.")
    eps = taujl.Imaginator.phase_fraction(im)
    if eps[1] != eps0[1]:
        if D is not None:
            D[~im] = 0.0
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
    }


def main():
    """Run a persistent request loop, reading file paths from a pipe fd."""
    ctrl_fd = int(sys.argv[1])
    ctrl_in = os.fdopen(ctrl_fd, "r")
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
        sys.stdout.write("done\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
