"""
Running several chains and measuring them, shared by Tests 5, 6 and 7.

Also holds small helpers that combine independent chains into one estimate
with an error bar and compare it with an exact value.
"""

import numpy as np

from .diagnostics import binning_error
from .lattice import exact_x2
from .metropolis import metropolis

Z_MAX = 3.5  # limit for statistical comparisons (plan, Section 5)


def run_chains(N, a, m, w, n_burn, n_keep, delta, seed, n_chains=4):
    """
    Run several chains: half start at x = 0, half from a wide random path.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    n_burn, n_keep : int
        Burn-in sweeps and kept sweeps per chain.
    delta : float
        Proposal half-width in the code's units.
    seed : int
        Base random seed.
    n_chains : int, optional
        Number of chains.

    Returns
    -------
    chains : ndarray, shape (n_chains, n_burn + n_keep + 1, N)
        Full chains, including the starting path and the burn-in.
    acc : ndarray, shape (n_chains,)
        Acceptance rate of each chain.
    starts : list of str
        "zero" or "wide" for each chain.
    """
    rng = np.random.default_rng(seed)
    width = 5 * np.sqrt(exact_x2(N, a, m, w))  # 5 times wider than the target
    chains, acc, starts = [], [], []
    for i in range(n_chains):
        wide = i >= n_chains // 2
        x0 = rng.normal(0, width, N) if wide else 0.0
        chain, rate = metropolis(
            N, a, m, w, n_burn + n_keep, delta, x0, seed + 1 + i, progress=False
        )
        chains.append(chain)
        acc.append(rate)
        starts.append("wide" if wide else "zero")
    return np.array(chains), np.array(acc), starts


def observables(chain):
    """
    Per-sweep values of the quantities we report.

    Parameters
    ----------
    chain : ndarray, shape (n, N)
        Paths after burn-in.

    Returns
    -------
    dict of ndarray
        "x": path average; "x2": C(0); "C1", "C2", "C4", "C8": C(j), averaged
        over t; "x4": <x^4>; "x_mid" and "x2_mid": x and x^2 at site N/2.
    """
    N = chain.shape[1]
    out = {"x": chain.mean(axis=1), "x2": (chain**2).mean(axis=1)}
    for j in (1, 2, 4, 8):
        out[f"C{j}"] = (chain * np.roll(chain, -j, axis=1)).mean(axis=1)
    out["x4"] = (chain**4).mean(axis=1)
    out["x_mid"] = chain[:, N // 2]
    out["x2_mid"] = chain[:, N // 2] ** 2
    return out


def mean_err(series):
    """
    Mean of one chain's per-sweep series and its binning error.

    Parameters
    ----------
    series : ndarray
        One value per sweep.

    Returns
    -------
    mean, err : float
        The mean and its error bar.
    """
    return float(np.mean(series)), float(binning_error(series)[1][-1])


def combine(per_chain):
    """
    Combine independent chains into one estimate.

    Parameters
    ----------
    per_chain : list of ndarray
        One per-sweep series for each chain.

    Returns
    -------
    mean : float
        Average of the chain means.
    err : float
        Its error, sqrt(sum of squared chain errors) / number of chains.
    """
    pairs = np.array([mean_err(s) for s in per_chain])
    mean = float(pairs[:, 0].mean())
    return mean, float(np.sqrt(np.sum(pairs[:, 1] ** 2)) / len(pairs))


def compare(label, mean, err, exact, limit=Z_MAX):
    """
    Compare a measured value with an exact one.

    Parameters
    ----------
    label : str
        Name of the quantity.
    mean, err : float
        Measured value and error bar.
    exact : float
        Expected value.
    limit : float, optional
        Largest allowed |z|.

    Returns
    -------
    line : str
        One line for the printout.
    ok : bool
        Whether |mean - exact| / err <= limit.
    """
    z = (mean - exact) / err
    ok = bool(abs(z) <= limit)
    line = (
        f"{label:26s} {mean:9.4f} ± {err:.4f}   expected {exact:8.4f}   "
        f"{z:+5.1f}σ  {'ok' if ok else 'FAIL'}"
    )
    return line, ok
