# Frequently asked questions

## Why doesn't `Ctrl-C` interrupt a running `tortuosity_fd` call?

Poromics sets `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` before importing
`juliacall`. This hands `SIGINT`, `SIGSEGV`, and `SIGBUS` off to Julia's
signal handlers so the runtime can recover from faults raised inside
JIT-compiled code or a GPU backend (on Darwin/arm64 this is what prevents
the ~60% SIGBUS rate during first-call Metal init — see
[#20](https://github.com/ma-sadeghi/poromics/issues/20)).

The tradeoff is that Python's own `SIGINT` handler does **not** run while
Julia owns the thread, so pressing `Ctrl-C` during a `tortuosity_fd` solve
won't raise `KeyboardInterrupt`. Use one of these instead:

- Let the solve finish — FDM runs typically return in seconds.
- Stop the whole Python process (`Ctrl-\`, or kill from another terminal).
- If you need Python-native signal handling (e.g. running in a long-lived
  notebook and you want a clean interrupt), override the default before
  importing `poromics`:

    ```python
    import os
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
    import poromics  # noqa: E402
    ```

    With `handle-signals=no` you regain `Ctrl-C`, but first-call Metal
    GPU initialization becomes unreliable on Darwin/arm64. This is a
    juliacall-wide limitation, not specific to Poromics.

The LBM solvers (`tortuosity_lbm`, `permeability_lbm`) are Python-driven
and honor `Ctrl-C` as usual.
