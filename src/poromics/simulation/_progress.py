# tqdm progress bar helpers for LBM solvers.
import math

from tqdm.auto import tqdm


def make_progress(n_steps, tol, label):
    """Create a tqdm progress bar for convergence tracking.

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
    pbar : tqdm
    """
    pbar = tqdm(total=100, desc=label, bar_format=(
        "{l_bar}{bar}| {n:.1f}% · {postfix} · {elapsed}"
    ))
    pbar.set_postfix_str("starting...")
    return pbar


def update_progress(pbar, step, ratio, tol, n_steps):
    """Update the progress bar based on convergence ratio.

    Uses log-scale estimation: if ratio started at ~1 and must reach
    tol, progress = log(ratio) / log(tol). This works because the
    residual decays roughly exponentially.

    Parameters
    ----------
    pbar : tqdm
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
    pbar.n = pct
    pbar.set_postfix_str(status)
    pbar.refresh()
