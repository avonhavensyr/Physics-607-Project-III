"""
Multi-walker test (validation plan, Test 11).

Several independent walkers are run on the main lattice, half starting at x = 0
and half from a wide random path. After burn-in we record, for every walker,
the fraction of sweeps in which a quantity q stays inside |q| < sigma_q, where
sigma_q is its exact standard deviation. If the sampler is right, every walker
spends the same fraction of time inside, and that fraction is the exact
probability erf(1/sqrt(2)) = 0.6827, because q is Gaussian.

Two quantities are used: x at the middle site, and the path average, which
moves the slowest. Error bars come from binning, so they include the
autocorrelation of each walker.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

from .diagnostics import binning_error
from .lattice import coefficients, exact_x2
from .metropolis import metropolis

Z_MAX = 3.5  # each walker against the exact fraction (plan, Section 5)
P_MIN = 1e-3  # walkers against each other (chi-square p-value)
P_INSIDE = float(special.erf(1 / np.sqrt(2)))  # exact P(|q| < sigma_q) = 0.6827


def run_walkers(N, a, m, w, n_walkers, n_burn, n_keep, delta, seed):
    """
    Run independent walkers and keep only the quantities Test 11 needs.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    n_walkers : int
        Number of walkers; the first half start at x = 0, the rest from a
        random path five times wider than the target distribution.
    n_burn, n_keep : int
        Burn-in sweeps and kept sweeps per walker.
    delta : float
        Proposal half-width in the code's units.
    seed : int
        Base random seed.

    Returns
    -------
    series : dict of ndarray, shape (n_walkers, n_keep)
        "x_mid": x at site N/2; "path average": the mean of the path.
    starts : list of str
        "zero" or "wide" for each walker.
    """
    rng = np.random.default_rng(seed)
    width = 5 * np.sqrt(exact_x2(N, a, m, w))  # 5 times wider than the target
    x_mid, x_bar, starts = [], [], []
    for i in range(n_walkers):
        wide = i >= n_walkers // 2
        x0 = rng.normal(0, width, N) if wide else 0.0
        chain, _ = metropolis(
            N, a, m, w, n_burn + n_keep, delta, x0, seed + 1 + i, progress=False
        )
        kept = chain[n_burn + 1 :]
        x_mid.append(kept[:, N // 2].copy())
        x_bar.append(kept.mean(axis=1))
        starts.append("wide" if wide else "zero")
    return {"x_mid": np.array(x_mid), "path average": np.array(x_bar)}, starts


def test11_walkers(
    N=64,
    a=0.25,
    m=1.0,
    w=1.0,
    n_walkers=8,
    n_burn=8_000,
    n_keep=40_000,
    Delta=2.5,
    seed=11,
):
    """
    Test 11: every walker spends the right fraction of time inside |q| < sigma_q.

    Parameters
    ----------
    N : int, optional
        Number of time slices.
    a, m, w : float, optional
        Lattice spacing, mass and frequency.
    n_walkers : int, optional
        Number of walkers.
    n_burn, n_keep : int, optional
        Burn-in sweeps and kept sweeps per walker.
    Delta : float, optional
        Step size in the plan's units; the code uses Delta / sqrt(A).
    seed : int, optional
        Base random seed.

    Returns
    -------
    dict
        Test result with "test", "name", "passed", "lines" and "figures".
    """
    A, _ = coefficients(a, m, w)
    delta = Delta / np.sqrt(A)  # the code's step for the plan's Delta
    series, starts = run_walkers(N, a, m, w, n_walkers, n_burn, n_keep, delta, seed)
    sigma = {
        "x_mid": np.sqrt(exact_x2(N, a, m, w)),  # one site: variance C(0)
        "path average": 1 / np.sqrt(N * a * m * w**2),  # variance 1/(N a m w^2)
    }
    lines, passed = [], True
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axi, (name, q) in zip(ax, series.items()):
        inside = (np.abs(q) < sigma[name]).astype(float)  # 1 when inside
        f = inside.mean(axis=1)
        err = np.array([binning_error(row)[1][-1] for row in inside])
        lines.append(
            f"{name}: interval |q| < {sigma[name]:.4f}, exact fraction {P_INSIDE:.4f}"
        )
        for i, start in enumerate(starts):
            z = (f[i] - P_INSIDE) / err[i]
            ok = bool(abs(z) <= Z_MAX)
            passed = passed and ok
            lines.append(
                f"  walker {i} ({start} start): {f[i]:.4f} ± {err[i]:.4f}  "
                f"{z:+5.1f}σ  {'ok' if ok else 'FAIL'}"
            )
        # walkers against each other: chi^2 around the weighted mean fraction
        wt = 1 / err**2
        f_bar = np.sum(wt * f) / np.sum(wt)
        chi2 = float(np.sum(wt * (f - f_bar) ** 2))
        p = float(stats.chi2.sf(chi2, n_walkers - 1))
        ok = p > P_MIN
        passed = passed and ok
        lines.append(
            f"  walkers agree: chi2 = {chi2:.1f} / {n_walkers - 1}, p = {p:.3f} "
            f"(limit {P_MIN})  {'ok' if ok else 'FAIL'}"
        )
        for i, start in enumerate(starts):
            color = "C0" if start == "zero" else "C1"
            label = f"{start} start" if starts.index(start) == i else None
            axi.errorbar(
                i, f[i], yerr=err[i], fmt="o", color=color, capsize=3, label=label
            )
        axi.axhline(P_INSIDE, color="k", ls="--", lw=1, label="exact 0.6827")
        axi.set(
            xlabel="walker",
            ylabel="fraction of sweeps inside",
            title=f"{name}: inside |q| < σ (walkers agree: p = {p:.2f})",
        )
        axi.legend(fontsize=8)
    return {
        "test": 11,
        "name": "Multi-walker test",
        "passed": bool(passed),
        "lines": lines,
        "figures": {"walkers": fig},
    }
