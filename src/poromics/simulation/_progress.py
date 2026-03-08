# Rich progress bar helpers for LBM solvers.
import math

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def make_progress(n_steps, tol, label):
    """Create a rich Progress bar and task for convergence tracking.

    Progress is estimated from how close the current residual is to the
    tolerance on a log scale: ``log(ratio) / log(tol)``. Since the
    residual decays roughly exponentially, this gives a near-linear
    progress estimate.

    Parameters
    ----------
    n_steps : int
        Maximum number of iterations (used in description).
    tol : float or None
        Convergence tolerance. If None, progress tracks step count.
    label : str
        Short label shown in the progress bar (e.g. "Diffusion").

    Returns
    -------
    progress : rich.progress.Progress
    task : TaskID
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TextColumn("·"),
        TextColumn("{task.fields[status]}"),
        TextColumn("·"),
        TimeElapsedColumn(),
    )
    task = progress.add_task(label, total=100, status="starting...")
    return progress, task


def update_progress(progress, task, step, ratio, tol, n_steps):
    """Update the progress bar based on convergence ratio.

    Uses log-scale estimation: if ratio started at ~1 and must reach
    tol, progress = log(ratio) / log(tol). This works because the
    residual decays roughly exponentially.

    Parameters
    ----------
    progress : rich.progress.Progress
    task : TaskID
    step : int
        Current iteration number.
    ratio : float
        Current residual ratio (delta / total).
    tol : float or None
        Target tolerance.
    n_steps : int
        Maximum iterations.
    """
    status = f"step {step}/{n_steps}  δ={ratio:.2e}"
    if tol is not None and tol < 1.0 and ratio > 0 and ratio < 1.0:
        log_tol = math.log(tol)
        log_ratio = math.log(ratio)
        pct = min(100.0, max(0.0, (log_ratio / log_tol) * 100))
    else:
        pct = (step / n_steps) * 100 if n_steps > 0 else 0
    progress.update(task, completed=pct, status=status)
