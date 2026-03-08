# API Reference

This page contains the API reference for the `poromics` package.

??? note "Metrics (`poromics`)"

    Convenience pipelines for computing transport properties. These are the main entry points for most users.

    ::: poromics

??? note "Simulation solvers (`poromics.simulation`)"

    Lower-level solver classes that accept physical SI units and expose intermediate state for custom post-processing.

    ::: poromics.simulation

??? note "Julia helpers (`poromics.julia_helpers`)"

    Helper functions for interacting with Julia and the `Tortuosity.jl` package. These are used internally by `tortuosity_fd`. Feel free to explore them if you're interested in the implementation details.

    ::: poromics.julia_helpers
