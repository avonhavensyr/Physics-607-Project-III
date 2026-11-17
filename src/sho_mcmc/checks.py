"""
Tests 1-5 of the validation plan, for the periodic-lattice code.

Each test returns a dict with "test", "name", "passed", "lines" (text to print)
and sometimes "figures". The limits are the plan's: 1e-10 for exact identities,
|z| <= 3.5 for statistical comparisons, KS p-value > 0.001, and "within 3 error
bars" for chains with different starting points.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

from .chains import combine, compare, mean_err, observables
from .diagnostics import autocorrelation, binning_error
from .injection import model_correlator
from .lattice import action, coefficients, exact_correlator, exact_values
from .metropolis import delta_action, expected_acceptance, metropolis

EXACT_TOL = 1e-10  # exact identities (plan, Tests 1 and 2)
P_MIN = 1e-3  # KS tests (plan, Test 4)


# ---------------- helpers ----------------
def matrix_K(N, A, B):
    """
    Matrix K of the action, S = x^T K x / 2, on the periodic lattice.

    Parameters
    ----------
    N : int
        Number of time slices.
    A, B : float
        Coefficients of the action.

    Returns
    -------
    ndarray, shape (N, N)
        2A on the diagonal and -B for each neighbour.
    """
    P = np.roll(np.eye(N), 1, axis=1)  # (P x)_t = x_{t+1}
    return 2 * A * np.eye(N) - B * (P + P.T)


def metropolis_loop(N, a, m, w, n_sweeps, delta, x0, seed=None):
    """
    Slow reference version of `metropolis` that updates one site at a time.

    It draws the same random numbers in the same order, so for the same seed it
    must give exactly the same chain. Used only by Test 1.

    Parameters
    ----------
    N : int
        Number of time slices (even).
    a, m, w : float
        Lattice spacing, mass and frequency.
    n_sweeps : int
        Number of sweeps.
    delta : float
        Proposal half-width.
    x0 : float or array_like
        Starting path.
    seed : int or None, optional
        Random seed.

    Returns
    -------
    ndarray, shape (n_sweeps + 1, N)
        The chain.
    """
    A, B = coefficients(a, m, w)
    rng = np.random.default_rng(seed)
    x = np.broadcast_to(np.asarray(x0, dtype=float), (N,)).copy()
    chain = np.empty((n_sweeps + 1, N))
    chain[0] = x
    for s in range(1, n_sweeps + 1):
        for parity in (0, 1):  # even sites, then odd sites
            shift = rng.uniform(-delta, delta, N)
            log_u = np.log(rng.random(N))
            for t in range(parity, N, 2):
                neighbours = x[t - 1] + x[(t + 1) % N]
                dS = shift[t] * (A * (shift[t] + 2 * x[t]) - B * neighbours)
                if log_u[t] < -dS:
                    x[t] += shift[t]
        chain[s] = x
    return chain


# ---------------- Test 1 ----------------
def test1_action_update(seed=1):
    """
    Test 1: the update's change of action, the action itself, and the sweep.

    Parameters
    ----------
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Test result.
    """
    rng = np.random.default_rng(seed)
    N, m, w = 64, 1.0, 1.0
    worst_dS = worst_quad = 0.0
    prob_ok = True
    for a in (0.25, 1.0):
        A, B = coefficients(a, m, w)
        K = matrix_K(N, A, B)
        for _ in range(500):  # 2 x 500 = 1000 random paths
            x = rng.normal(0, 1, N)
            shift = rng.uniform(-2, 2, N)
            t = rng.integers(N)
            moved = x.copy()
            moved[t] += shift[t]
            full = action(moved, a, m, w) - action(x, a, m, w)
            fast = delta_action(x, shift, A, B)[t]
            worst_dS = max(worst_dS, abs(full - fast))
            worst_quad = max(worst_quad, abs(0.5 * x @ K @ x - action(x, a, m, w)))
            p = np.exp(min(0.0, -fast))  # acceptance probability min(1, e^-dS)
            prob_ok = prob_ok and 0.0 <= p <= 1.0

    fast_chain, _ = metropolis(8, 0.5, m, w, 200, 1.0, 0.0, seed)
    loop_chain = metropolis_loop(8, 0.5, m, w, 200, 1.0, 0.0, seed)
    loop_diff = float(np.max(np.abs(fast_chain - loop_chain)))
    passed = worst_dS < EXACT_TOL and worst_quad < EXACT_TOL and prob_ok
    passed = passed and loop_diff == 0.0
    lines = [
        f"1000 random paths: max |dS - (S_new - S_old)| = {worst_dS:.1e}"
        f" (limit {EXACT_TOL:g})",
        f"1000 random paths: max |x^T K x / 2 - S(x)| = {worst_quad:.1e}"
        f" (limit {EXACT_TOL:g})",
        f"acceptance probability always between 0 and 1: {prob_ok}",
        f"checkerboard sweep vs site-by-site loop (same random numbers): "
        f"max difference {loop_diff:g} (must be exactly 0)",
    ]
    name = "Action and update"
    return {"test": 1, "name": name, "passed": bool(passed), "lines": lines}


# ---------------- Test 2 ----------------
def test2_exact_covariance(N=64, a=0.25, m=1.0, w=1.0):
    """
    Test 2: the exact covariance used by every other test.

    Parameters
    ----------
    N : int, optional
        Number of time slices.
    a, m, w : float, optional
        Lattice spacing, mass and frequency.

    Returns
    -------
    dict
        Test result.
    """
    A, B = coefficients(a, m, w)
    K = matrix_K(N, A, B)
    eig = np.sort(np.linalg.eigvalsh(K))
    expected = np.sort(2 * A - 2 * B * np.cos(2 * np.pi * np.arange(N) / N))
    eig_diff = float(np.max(np.abs(eig - expected)))
    C_inv = np.linalg.inv(K)[0]  # C(j) = (K^-1)_{0j}
    d_sum = float(np.max(np.abs(C_inv - exact_correlator(N, a, m, w))))
    closed = model_correlator(np.arange(N), N, a, m, w)
    d_closed = float(np.max(np.abs(C_inv - closed)))
    symmetric = bool(np.array_equal(K, K.T))
    posdef = bool(eig[0] > 0)
    passed = symmetric and posdef and max(eig_diff, d_sum, d_closed) < EXACT_TOL
    values = ", ".join(f"C({j}) = {C_inv[j]:.4f}" for j in (0, 1, 2, 4, 8))
    lines = [
        f"N={N}, a={a}: K symmetric: {symmetric}; smallest eigenvalue "
        f"{eig[0]:.4f} > 0: {posdef}",
        f"eigenvalues vs 2A - 2B cos(2 pi k/N): max difference {eig_diff:.1e}",
        f"C(j) from NumPy's inverse vs eigenvalue sum: {d_sum:.1e}; "
        f"vs closed form: {d_closed:.1e}",
        f"values: {values}",
    ]
    name = "Exact covariance"
    return {"test": 2, "name": name, "passed": bool(passed), "lines": lines}


# ---------------- Test 3 ----------------
def _ar1(phi, n, rng):
    """
    AR(1) series x_t = phi x_{t-1} + sqrt(1 - phi^2) e_t, started in equilibrium.

    Parameters
    ----------
    phi : float
        Correlation between neighbours.
    n : int
        Length.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray, shape (n,)
        The series, with variance 1 and tau_int = (1 + phi) / (2 (1 - phi)).
    """
    e = rng.normal(size=n) * np.sqrt(1 - phi**2)
    e[0] = rng.normal()  # x_0 ~ N(0, 1)
    return signal.lfilter([1.0], [1.0, -phi], e)


def test3_error_bars(seed=3, n=2**22):
    """
    Test 3: the error-bar code on AR(1) data with known answers, and R-hat.

    Parameters
    ----------
    seed : int, optional
        Random seed.
    n : int, optional
        Length of each AR(1) series.

    Returns
    -------
    dict
        Test result.
    """
    rng = np.random.default_rng(seed)
    lines, passed = [], True
    for phi in (0.5, 0.9, 0.99):
        x = _ar1(phi, n, rng)
        tau_exact = (1 + phi) / (2 * (1 - phi))
        tau = autocorrelation(x)[1]
        sizes, errs, _ = binning_error(x)
        k = min(int(np.searchsorted(sizes, 20 * tau)), len(sizes) - 1)  # >= 20 tau
        # exact error of the mean of a unit-variance AR(1) series of length n
        var_mean = (1 + phi) / (1 - phi) - 2 * phi * (1 - phi**n) / (n * (1 - phi) ** 2)
        err_exact = np.sqrt(var_mean / n)
        ok_tau = abs(tau / tau_exact - 1) < 0.10
        ok_err = abs(errs[k] / err_exact - 1) < 0.20
        passed = passed and ok_tau and ok_err
        lines.append(
            f"phi={phi}: tau_int {tau:7.2f} vs exact {tau_exact:6.2f} "
            f"({100 * (tau / tau_exact - 1):+.1f}%, limit 10%)  "
            f"error (blocks of {sizes[k]}) {errs[k]:.2e} vs exact {err_exact:.2e} "
            f"({100 * (errs[k] / err_exact - 1):+.1f}%, limit 20%)"
        )
    try:  # the R-hat code is in convergence.py (Test 7)
        from .convergence import rhat
    except ImportError:
        lines.append("R-hat check not run: convergence.py is not added yet")
        name = "Error-bar code"
        return {"test": 3, "name": name, "passed": False, "lines": lines}
    chains = rng.normal(size=(4, 10_000))
    r_same = rhat(chains)
    shifted = chains.copy()
    shifted[0] += 0.5  # move one chain by half a standard deviation
    r_shift = rhat(shifted)
    ok_r = r_same < 1.01 and r_shift > 1.01
    passed = passed and ok_r
    lines.append(
        f"R-hat: 4 matching chains {r_same:.4f} (must be < 1.01); "
        f"one chain shifted {r_shift:.4f} (must be > 1.01)"
    )
    return {"test": 3, "name": "Error-bar code", "passed": bool(passed), "lines": lines}


# ---------------- Test 4 ----------------
def test4_small_lattice(seed=4, n_sweeps=200_000, burn=1_000, delta=1.0):
    """
    Test 4: the smallest lattice, N = 2 at a = 1, where everything is known.

    x0 + x1 and x0 - x1 are independent Gaussians with variances 1/(A - B) and
    1/(A + B) (2 and 0.4 for a = 1). N = 3 from the original plan cannot run,
    because the checkerboard update needs an even N.

    Parameters
    ----------
    seed : int, optional
        Random seed.
    n_sweeps, burn : int, optional
        Sweeps and burn-in sweeps.
    delta : float, optional
        Proposal half-width in the code's units.

    Returns
    -------
    dict
        Test result.
    """
    N, a, m, w = 2, 1.0, 1.0, 1.0
    A, B = coefficients(a, m, w)
    chain, _, hist = metropolis(N, a, m, w, n_sweeps, delta, 0.0, seed, history=True)
    x = chain[burn + 1 :]
    s, d = x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]
    var_s, var_d = 1 / (A - B), 1 / (A + B)

    # the KS test assumes independent draws, so thin by 5 tau_int
    tau = max(autocorrelation(v)[1] for v in (s, d, s**2, d**2))
    thin = int(np.ceil(5 * tau))
    p_s = stats.kstest(s[::thin] / np.sqrt(var_s), "norm").pvalue
    p_d = stats.kstest(d[::thin] / np.sqrt(var_d), "norm").pvalue
    passed = p_s > P_MIN and p_d > P_MIN
    lines = [
        f"KS test vs exact Gaussian (every {thin}th sweep, {len(s[::thin])} draws): "
        f"x0+x1 p = {p_s:.3f}, x0-x1 p = {p_d:.3f} (limit {P_MIN})"
    ]
    moments = [
        ("var(x0+x1)", s**2, var_s),
        ("var(x0-x1)", d**2, var_d),
        ("<(x0+x1)^4>", s**4, 3 * var_s**2),
        ("<(x0-x1)^4>", d**4, 3 * var_d**2),
        ("<(x0+x1)(x0-x1)>", s * d, 0.0),
        ("acceptance", hist[burn:], float(expected_acceptance(delta, a, m, w))),
    ]
    for label, series, exact in moments:
        line, ok = compare(label, *mean_err(series), exact)
        lines.append(line)
        passed = passed and ok

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    panels = ((ax[0], s, var_s, "x0 + x1"), (ax[1], d, var_d, "x0 - x1"))
    for axi, v, var, label in panels:
        axi.hist(v, bins=100, density=True, alpha=0.6, label="MCMC")
        grid = np.linspace(-5 * np.sqrt(var), 5 * np.sqrt(var), 300)
        axi.plot(grid, stats.norm.pdf(grid, scale=np.sqrt(var)), "k", label="exact")
        axi.set(xlabel=label, ylabel="density", title=f"N = 2: {label}")
        axi.legend()
    return {
        "test": 4,
        "name": "Smallest lattice (N = 2)",
        "passed": bool(passed),
        "lines": lines,
        "figures": {"small_lattice": fig},
    }


# ---------------- Test 5 ----------------
def test5_known_answers(chains, n_burn, N, a, m, w):
    """
    Test 5: measured averages on the main lattice against the exact values.

    Every quantity is averaged over all time slices (the lattice is
    translation invariant); there are no end sites on a periodic lattice.
    The energies come from x^2 and C(1) of each path: V = m w^2 x^2 / 2,
    T = 1/(2a) - m <(x_{t+1} - x_t)^2> / (2 a^2) = 1/(2a) - m (x^2 - C(1)) / a^2
    (primitive estimator), E = T + V, and the virial estimator E_vir = 2 V.

    Parameters
    ----------
    chains : ndarray, shape (n_chains, n_sweeps + 1, N)
        Output of `sho_mcmc.chains.run_chains`.
    n_burn : int
        Burn-in sweeps to drop.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    dict
        Test result.
    """
    obs = [observables(c[n_burn + 1 :]) for c in chains]
    for o in obs:  # energies per sweep, from x^2 and C(1) of each path
        o["V"] = 0.5 * m * w**2 * o["x2"]
        o["T"] = 1 / (2 * a) - m * (o["x2"] - o["C1"]) / a**2
        o["E"] = o["T"] + o["V"]
        o["E_vir"] = 2 * o["V"]
    C = exact_correlator(N, a, m, w)
    lat, _ = exact_values(N, a, m, w)
    expected = {
        "<x>": ("x", 0.0),
        "<x^2> = C(0)": ("x2", C[0]),
        "C(1)": ("C1", C[1]),
        "C(2)": ("C2", C[2]),
        "C(4)": ("C4", C[4]),
        "C(8)": ("C8", C[8]),
        "<x^4> = 3 C(0)^2": ("x4", 3 * C[0] ** 2),
        "<V>": ("V", lat["V"]),
        "<T> (primitive)": ("T", lat["T_prim"]),
        "E (primitive)": ("E", lat["E_prim"]),
        "E (virial)": ("E_vir", lat["E_vir"]),
    }
    lines, passed = [], True
    for label, (key, exact) in expected.items():
        line, ok = compare(label, *combine([o[key] for o in obs]), exact)
        lines.append(line)
        passed = passed and ok
    name = "Known answers (N = 64)"
    return {"test": 5, "name": name, "passed": bool(passed), "lines": lines}
