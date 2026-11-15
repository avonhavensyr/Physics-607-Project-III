"""
Test 8 of the validation plan, and the choice of the step size.

The plan measures the step in units where one site, given its neighbours, has
variance 1/2. In the code's units that step is delta = Delta / sqrt(A), with
A = m/a + a m w^2 / 2. With this conversion the average acceptance is the plan's
Eq. (8) and does not depend on the lattice spacing.
"""

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import autocorrelation, binning_error
from .lattice import coefficients
from .metropolis import expected_acceptance, metropolis

Z_MAX = 3.5  # statistical comparisons (plan, Section 5)
DELTAS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # pilot step sizes (plan, Section 4)
ACC_RANGE = (0.35, 0.65)  # acceptance range allowed by the plan


def test8_acceptance(seed=8, n_sweeps=40_000, burn=1_000):
    """
    Test 8: measured acceptance against Eq. (8), plus the step-size study.

    Every pilot step is run at a = 0.25 (N = 64) and a = 1 (N = 16). At
    a = 0.25 the runs are long enough to measure tau_int of the path average
    and of x^2, which gives the effective samples per sweep for each step.

    Parameters
    ----------
    seed : int, optional
        Base random seed.
    n_sweeps : int, optional
        Sweeps per run at a = 0.25 (an eighth of that at a = 1).
    burn : int, optional
        Burn-in sweeps.

    Returns
    -------
    dict
        Test result, including "best_Delta" and the figures.
    """
    lines, rows, passed = [], [], True
    runs = ((0.25, 64, n_sweeps), (1.0, 16, n_sweeps // 8))  # (a, N, sweeps)
    for k, (a, N, sweeps) in enumerate(runs):
        A, _ = coefficients(a, 1.0, 1.0)
        for i, D in enumerate(DELTAS):
            delta = D / np.sqrt(A)  # the code's step for the plan's Delta
            run_seed = seed + 100 * k + i
            chain, _, hist = metropolis(
                N,
                a,
                1.0,
                1.0,
                sweeps,
                delta,
                0.0,
                run_seed,
                history=True,
                progress=False,
            )
            h = hist[burn:]
            acc, err = h.mean(), binning_error(h)[1][-1]
            exp = float(expected_acceptance(delta, a, 1.0, 1.0))
            z = (acc - exp) / err
            ok = bool(abs(z) <= Z_MAX)
            passed = passed and ok
            row = {"a": a, "Delta": D, "acc": acc, "err": err, "exp": exp}
            if a == 0.25:
                x = chain[burn + 1 :]
                row["tau"] = max(
                    autocorrelation(x.mean(axis=1))[1],
                    autocorrelation((x**2).mean(axis=1))[1],
                )
            rows.append(row)
            lines.append(
                f"a={a:<4} Delta={D:<3} (delta={delta:.3f}): acceptance {acc:.4f} ± "
                f"{err:.4f}, Eq. (8) {exp:.4f}, {z:+5.1f}σ  {'ok' if ok else 'FAIL'}"
            )

    # step-size study: most effective samples per sweep inside the allowed range
    study = [r for r in rows if r["a"] == 0.25]
    allowed = [r for r in study if ACC_RANGE[0] <= r["acc"] <= ACC_RANGE[1]]
    best = max(allowed, key=lambda r: 1 / r["tau"])
    lines.append("step-size study at a = 0.25 (slowest of path average and x^2):")
    for r in study:
        lines.append(
            f"  Delta={r['Delta']:<3}  acceptance {r['acc']:.3f}  tau_int "
            f"{r['tau']:6.1f} sweeps  effective samples per 1000 sweeps "
            f"{1000 / (2 * r['tau']):5.1f}"
        )
    lines.append(
        f"best Delta with acceptance in {ACC_RANGE}: {best['Delta']} "
        f"(the plan expected about 2.5)"
    )

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    grid = np.linspace(0.3, 4.2, 200)
    ax[0].plot(grid, _eq8(grid), "k", label="Eq. (8)")
    for a, marker in ((0.25, "o"), (1.0, "s")):
        sel = [r for r in rows if r["a"] == a]
        ax[0].errorbar(
            [r["Delta"] for r in sel],
            [r["acc"] for r in sel],
            yerr=[r["err"] for r in sel],
            fmt=marker,
            label=f"a = {a}",
        )
    ax[0].axhspan(*ACC_RANGE, color="C2", alpha=0.1, label="allowed range")
    ax[0].set(
        xlabel=r"$\Delta$ (plan units)",
        ylabel="acceptance",
        title="Acceptance against Eq. (8)",
    )
    ax[0].legend()
    deltas = [r["Delta"] for r in study]
    ax[1].plot(deltas, [1000 / (2 * r["tau"]) for r in study], "o-")
    chosen = f"chosen Δ = {best['Delta']}"
    ax[1].axvline(best["Delta"], color="r", ls="--", label=chosen)
    ax[1].set(
        xlabel=r"$\Delta$ (plan units)",
        ylabel="effective samples / 1000 sweeps",
        title="Step-size study (a = 0.25, N = 64)",
    )
    ax[1].legend()
    return {
        "test": 8,
        "name": "Acceptance and step size",
        "passed": bool(passed),
        "lines": lines,
        "best_Delta": best["Delta"],
        "figures": {"acceptance": fig},
    }


def _eq8(D):
    """
    Eq. (8) of the plan as a function of the step in the plan's units.

    Parameters
    ----------
    D : float or ndarray
        Step size Delta in the plan's units.

    Returns
    -------
    float or ndarray
        Average acceptance rate.
    """
    A, _ = coefficients(1.0, 1.0, 1.0)
    return expected_acceptance(np.asarray(D) / np.sqrt(A), 1.0, 1.0, 1.0)
