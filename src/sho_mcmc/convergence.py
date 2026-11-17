"""
Convergence checks with several chains (validation plan, Test 7).

R-hat is the rank-normalized split R-hat of Vehtari et al. (2021): every chain
is split in half, the draws are replaced by normal scores of their ranks, and
R-hat is computed both for the draws and for their distance from the median,
which catches chains that differ in spread. The effective sample size is
n / (2 tau_int) for each chain, summed over chains, as in Section 5 of the plan.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .chains import observables
from .diagnostics import autocorrelation

RHAT_MAX = 1.01  # pass limit from the plan
ESS_MIN = 400  # pass limit from the plan (total over all chains)


def _rhat_basic(chains):
    """
    Gelman-Rubin R-hat.

    Parameters
    ----------
    chains : ndarray, shape (m, n)
        m chains of n draws.

    Returns
    -------
    float
        sqrt(((n-1)/n W + B/n) / W), with W the mean within-chain variance and
        B/n the variance of the chain means.
    """
    n = chains.shape[1]
    W = chains.var(axis=1, ddof=1).mean()
    B = n * chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def _rank_normalize(x):
    """
    Replace draws by normal scores of their ranks over all chains.

    Parameters
    ----------
    x : ndarray, shape (m, n)
        Draws.

    Returns
    -------
    ndarray, shape (m, n)
        Phi^{-1}((r - 3/8) / (S + 1/4)), with r the rank and S the number of
        draws (Blom's formula).
    """
    r = stats.rankdata(x).reshape(x.shape)
    return stats.norm.ppf((r - 3 / 8) / (x.size + 1 / 4))


def rhat(chains):
    """
    Rank-normalized split R-hat.

    Parameters
    ----------
    chains : array_like, shape (m, n)
        m chains of n draws of one quantity, burn-in removed.

    Returns
    -------
    float
        The larger of the bulk and tail R-hat; near 1 when the chains agree.
    """
    x = np.asarray(chains, dtype=float)
    half = x.shape[1] // 2
    split = np.concatenate([x[:, :half], x[:, half : 2 * half]])  # 2m chains
    bulk = _rhat_basic(_rank_normalize(split))
    tail = _rhat_basic(_rank_normalize(np.abs(split - np.median(split))))
    return max(bulk, tail)


def ess(chains):
    """
    Effective sample size, summed over chains.

    Parameters
    ----------
    chains : array_like, shape (m, n)
        m chains of n draws of one quantity, burn-in removed.

    Returns
    -------
    float
        sum over chains of n / (2 tau_int).
    """
    x = np.asarray(chains, dtype=float)
    return float(sum(len(c) / (2 * autocorrelation(c)[1]) for c in x))


def test7_convergence(chains, n_burn, starts):
    """
    Test 7: R-hat < 1.01 and ESS > 400 for every reported quantity.

    Parameters
    ----------
    chains : ndarray, shape (m, n_sweeps + 1, N)
        Output of `sho_mcmc.chains.run_chains`.
    n_burn : int
        Burn-in sweeps to drop.
    starts : list of str
        "zero" or "wide" for each chain, for the trace plots.

    Returns
    -------
    dict
        Test result with "test", "name", "passed", "lines" and "figures".
    """
    obs = [observables(c[n_burn + 1 :]) for c in chains]
    lines, passed = [], True
    for name in obs[0]:
        series = np.array([o[name] for o in obs])
        r, e = rhat(series), ess(series)
        ok = r < RHAT_MAX and e > ESS_MIN
        passed = passed and ok
        lines.append(
            f"{name:7s} R-hat {r:.4f} (limit {RHAT_MAX})   "
            f"ESS {e:7.0f} (limit {ESS_MIN})   {'ok' if ok else 'FAIL'}"
        )

    # trace plots of the full chains, so the burn-in from both starts is visible
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    N = chains.shape[2]
    traces = [
        (lambda c: c[:, N // 2], f"$x_{{{N // 2}}}$"),
        (lambda c: c[:, N // 2] ** 2, f"$x_{{{N // 2}}}^2$"),
        (lambda c: c.mean(axis=1), r"path average $\bar{x}$"),
    ]
    for axi, (f, label) in zip(ax, traces):
        for i, (c, start) in enumerate(zip(chains, starts)):
            axi.plot(f(c), lw=0.3, alpha=0.7, label=f"chain {i} ({start} start)")
        axi.axvline(n_burn, color="k", ls="--", lw=1)
        axi.set(xlabel="sweep", ylabel=label, title=f"Trace of {label}")
    ax[0].legend(fontsize=7)
    return {
        "test": 7,
        "name": "Convergence (R-hat, ESS, traces)",
        "passed": bool(passed),
        "lines": lines,
        "figures": {"traces": fig},
    }
