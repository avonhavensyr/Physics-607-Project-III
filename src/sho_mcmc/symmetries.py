"""
Symmetries of the distribution and chains with different starts (Test 6).

On the periodic lattice the distribution does not change under x -> -x, under
time reversal x_t -> x_{N-t}, or under time shifts x_t -> x_{t+s}. Each
symmetry gives an exact prediction that the chains must reproduce. There are no
degeneracies: the matrix of the action is positive definite, so the
distribution has a single peak at x = 0.
"""

import numpy as np

from .chains import combine, compare, observables
from .lattice import exact_x2


def test6_symmetries(chains, n_burn, starts, N, a, m, w):
    """
    Test 6: symmetries of the distribution, and chains with different starts.

    Parameters
    ----------
    chains : ndarray, shape (n_chains, n_sweeps + 1, N)
        Output of `sho_mcmc.chains.run_chains`.
    n_burn : int
        Burn-in sweeps to drop.
    starts : list of str
        "zero" or "wide" for each chain.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    dict
        Test result with "test", "name", "passed" and "lines".
    """
    kept = [c[n_burn + 1 :] for c in chains]
    C0 = exact_x2(N, a, m, w)
    lines, passed = [], True
    quarter = (N // 4, N // 2, 3 * N // 4)
    for t in quarter:  # x -> -x: the mean is zero at every site
        line, ok = compare(f"<x_{t}>", *combine([c[:, t] for c in kept]), 0.0)
        lines.append(line)
        passed = passed and ok
    j0 = N // 2  # time reversal: <x_j0 x_j0+4> = <x_j0 x_j0-4>
    diff = [c[:, j0] * (c[:, (j0 + 4) % N] - c[:, j0 - 4]) for c in kept]
    label = f"<x_{j0}x_{j0 + 4}> - <x_{j0}x_{j0 - 4}>"
    line, ok = compare(label, *combine(diff), 0.0)
    lines.append(line)
    passed = passed and ok
    for t in (0,) + quarter:  # time shift: the same variance at every site
        line, ok = compare(f"<x_{t}^2>", *combine([c[:, t] ** 2 for c in kept]), C0)
        lines.append(line)
        passed = passed and ok

    obs = [observables(c) for c in kept]  # starting points: zero vs wide
    for key, name in (("x2", "<x^2>"), ("C4", "C(4)")):
        m0, e0 = combine([o[key] for o, s in zip(obs, starts) if s == "zero"])
        m1, e1 = combine([o[key] for o, s in zip(obs, starts) if s == "wide"])
        z = (m0 - m1) / np.hypot(e0, e1)
        ok = bool(abs(z) <= 3)
        passed = passed and ok
        lines.append(
            f"{name}: zero start {m0:.4f} ± {e0:.4f}, wide start {m1:.4f} ± "
            f"{e1:.4f}: {z:+.1f}σ (limit 3)  {'ok' if ok else 'FAIL'}"
        )
    name = "Symmetries and starts"
    return {"test": 6, "name": name, "passed": bool(passed), "lines": lines}
