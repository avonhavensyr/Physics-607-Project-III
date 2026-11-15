"""
Injection-recovery test for the lattice oscillator (validation plan, Test 10).

Known values of m and w are "injected" into the simulation, paths are sampled
with our Metropolis code, and m and w are recovered from the samples alone. If
the sampler is right, the recovered values agree with the injected ones.

On the periodic lattice the correlator C(j) = <x_t x_{t+j}> is known exactly
for 0 <= j <= N,

    C(j) = cosh(mu (N/2 - j)) / (2 (m/a) sinh(mu) sinh(mu N/2)),
    cosh(mu) = 1 + (a w)^2 / 2,

so the measured C(j) is fitted with this formula, with m and w as the two free
parameters: the decay rate gives w and the overall size gives m. The formula is
written out here from the lattice action instead of being taken from
`sho_mcmc.lattice`, so a bug there cannot cancel out of the comparison.

Pass (limits from the validation plan): |recovered - injected| <= 3.5 error
bars for both m and w, and fit p-value > 0.001.

Negative control: one more run is sampled with w = 1.1 but compared with
w = 1. The test must flag it; otherwise it is too weak to be useful.

Run with ``sho-mcmc-inject`` (options: ``sho-mcmc-inject --help``).
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

from .diagnostics import autocorrelation
from .lattice import exact_x2
from .metropolis import metropolis

Z_MAX = 3.5  # pass limit on |recovered - injected| / error
P_MIN = 1e-3  # pass limit on the fit p-value
MIN_BLOCK = 20  # blocks must be at least this many tau_int long (from the plan)

# Injected (N, a, m, w); all have T = N a = 16 like the original plan. The last
# two use m != 1 and w != 1: with m = w = 1 some mistakes (for example a missing
# factor of m in the potential) would not show up.
CASES = [
    (16, 1.0, 1.0, 1.0),
    (32, 0.5, 1.0, 1.0),
    (64, 0.25, 1.0, 1.0),
    (32, 0.5, 2.0, 0.5),
    (32, 0.5, 0.5, 2.0),
]


# ---------------- exact lattice results ----------------
def lattice_mu(a, w):
    """
    Decay rate of the correlator per lattice step.

    Parameters
    ----------
    a : float or ndarray
        Lattice spacing.
    w : float or ndarray
        Oscillator frequency.

    Returns
    -------
    float or ndarray
        mu with cosh(mu) = 1 + (a w)^2 / 2, i.e. mu = 2 asinh(a w / 2).
        The lattice energy gap is mu / a.
    """
    return 2 * np.arcsinh(a * w / 2)


def model_correlator(j, N, a, m, w):
    """
    Exact correlator C(j) = <x_t x_{t+j}> on the periodic lattice.

    Parameters
    ----------
    j : int or ndarray
        Separation(s), 0 <= j <= N.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    float or ndarray
        C(j) for the action S = sum_t [m/(2a) (x_{t+1}-x_t)^2 + a m w^2 x_t^2 / 2].
    """
    mu = lattice_mu(a, w)
    norm = 2 * (m / a) * np.sinh(mu) * np.sinh(mu * N / 2)
    return np.cosh(mu * (N / 2 - j)) / norm


def check_model(N, a, m, w):
    """
    Test the test: check the closed-form correlator before using it.

    It must match a direct sum over the eigenvalues of the action, and C(0)
    must match `sho_mcmc.lattice.exact_x2`.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Raises
    ------
    AssertionError
        If either comparison fails.
    """
    theta = 2 * np.pi * np.arange(N) / N
    lam = 2 * (m / a + a * m * w**2 / 2) - 2 * (m / a) * np.cos(theta)  # eigenvalues
    direct = np.array([np.mean(np.cos(theta * j) / lam) for j in range(N + 1)])
    closed = model_correlator(np.arange(N + 1), N, a, m, w)
    assert np.allclose(closed, direct, rtol=1e-10), "closed-form correlator is wrong"
    x2 = exact_x2(N, a, m, w)
    assert np.isclose(closed[0], x2, rtol=1e-10), "exact_x2 disagrees"


# ---------------- measurement and fit ----------------
def correlator_per_sweep(chain):
    """
    Measured correlator, one row per sweep.

    Parameters
    ----------
    chain : ndarray, shape (n, N)
        Paths after burn-in.

    Returns
    -------
    ndarray, shape (n, N//2 + 1)
        C(j) = (1/N) sum_t x_t x_{t+j} for j = 0, ..., N/2. The FFT gives the
        circular correlation, which is exactly right on a periodic lattice.
    """
    N = chain.shape[1]
    f = np.fft.rfft(chain, axis=1)
    return np.fft.irfft(np.abs(f) ** 2, n=N, axis=1)[:, : N // 2 + 1] / N


def fit_window(C):
    """
    Choose the fit range from the data, without using the injected values.

    The rough decay rate comes from C(0) + C(2) = 2 cosh(mu) C(1), and the fit
    uses times between 0.5 and 3 decay times, as in the original plan.

    Parameters
    ----------
    C : ndarray
        Mean correlator for j = 0, ..., N/2.

    Returns
    -------
    js : ndarray of int
        Separations used in the fit (at least 3).
    mu0 : float
        Rough decay rate per lattice step.

    Raises
    ------
    RuntimeError
        If the decay rate cannot be estimated (too few sweeps).
    """
    mu0 = np.arccosh((C[0] + C[2]) / (2 * C[1]))
    if not np.isfinite(mu0) or mu0 <= 0:
        raise RuntimeError("could not estimate the decay rate; run more sweeps")
    j_end = len(C) - 1
    j_min = max(1, int(np.ceil(0.5 / mu0)))
    j_max = min(j_end, max(int(np.floor(3 / mu0)), j_min + 2))
    return np.arange(j_min, j_max + 1), mu0


def fit_m_w(blocks, js, N, a, guess):
    """
    Correlated least-squares fit of the block-averaged correlator.

    Parameters
    ----------
    blocks : ndarray, shape (n_blocks, N//2 + 1)
        Correlator averaged within blocks of consecutive sweeps.
    js : ndarray of int
        Separations used in the fit.
    N : int
        Number of time slices.
    a : float
        Lattice spacing (known, not fitted).
    guess : sequence of float
        Starting values (m, w).

    Returns
    -------
    params : ndarray
        Recovered (m, w).
    cov : ndarray, shape (2, 2)
        Covariance of (m, w).
    chi2 : float
        Chi-square at the minimum, with the Hartlap correction.
    dof : int
        Degrees of freedom.
    p : float
        Fit p-value.
    """
    nb, k = blocks.shape[0], len(js)
    y = blocks[:, js].mean(axis=0)
    cov_y = np.cov(blocks[:, js], rowvar=False) / nb  # covariance of the mean
    chol = np.linalg.cholesky(cov_y)

    def residuals(p):
        # whitened residuals: their sum of squares is the correlated chi^2
        return np.linalg.solve(chol, y - model_correlator(js, N, a, *p))

    # w enters only as w^2, so w and -w fit equally well; we keep w > 0
    res = optimize.least_squares(residuals, guess, bounds=([1e-9, 1e-9], np.inf))
    cov = np.linalg.inv(res.jac.T @ res.jac)
    # the inverse of an estimated covariance is biased; Hartlap factor corrects chi^2
    chi2 = 2 * res.cost * (nb - k - 2) / (nb - 1)
    dof = k - 2
    return res.x, cov, chi2, dof, stats.chi2.sf(chi2, dof)


def run_case(N, a, m, w, sweeps, burn, delta, n_blocks, seed):
    """
    Inject (m, w), sample, recover (m, w) and compare.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing and the injected mass and frequency.
    sweeps, burn : int
        Number of sweeps and of burn-in sweeps.
    delta : float
        Proposal half-width.
    n_blocks : int
        Number of blocks for the error analysis.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Settings, recovered values, errors, pulls, fit quality and pass/fail.
    """
    check_model(N, a, m, w)
    chain, acc = metropolis(N, a, m, w, sweeps, delta, 0.0, seed, progress=False)
    C = correlator_per_sweep(chain[burn:])
    del chain
    C_mean = C.mean(axis=0)
    js, mu0 = fit_window(C_mean)

    # blocks of consecutive sweeps; check that they are long enough
    tau = max(autocorrelation(C[:, j])[1] for j in js)
    length = len(C) // n_blocks
    blocks = C[: n_blocks * length].reshape(n_blocks, length, -1).mean(axis=1)
    C_err = blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)

    # starting guess from the data: w from mu0, m from C(0)
    w0 = 2 / a * np.sinh(mu0 / 2)
    m0 = a / (2 * C_mean[0] * np.sinh(mu0) * np.tanh(mu0 * N / 2))
    (m_rec, w_rec), cov, chi2, dof, p = fit_m_w(blocks, js, N, a, [m0, w0])
    m_err, w_err = np.sqrt(np.diag(cov))

    z_m, z_w = (m_rec - m) / m_err, (w_rec - w) / w_err
    gap = lattice_mu(a, w_rec) / a  # lattice energy gap E1 - E0
    gap_err = w_err / np.sqrt(1 + (a * w_rec / 2) ** 2)
    passed = abs(z_m) <= Z_MAX and abs(z_w) <= Z_MAX and p > P_MIN
    return {
        "N": N,
        "a": a,
        "m": m,
        "w": w,
        "seed": seed,
        "acc": acc,
        "m_rec": m_rec,
        "m_err": m_err,
        "w_rec": w_rec,
        "w_err": w_err,
        "z_m": z_m,
        "z_w": z_w,
        "gap": gap,
        "gap_err": gap_err,
        "gap_lattice": lattice_mu(a, w) / a,
        "chi2": chi2,
        "dof": dof,
        "p": p,
        "js": js.tolist(),
        "tau": tau,
        "block_over_tau": length / tau,
        "C": C_mean.tolist(),
        "C_err": C_err.tolist(),
        "passed": bool(passed),
    }


# ---------------- report ----------------
def print_table(results):
    """
    Print one line per run, warnings, and a summary of the pulls.

    Parameters
    ----------
    results : list of dict
        Outputs of `run_case`.
    """
    print(
        f"{'N':>3s} {'a':>5s} {'m':>5s} {'w':>5s} {'acc':>5s} | "
        f"{'m_rec':>7s} {'± err':>7s} {'pull':>6s} | "
        f"{'w_rec':>7s} {'± err':>7s} {'pull':>6s} | "
        f"{'gap/w':>6s} {'(lat)':>7s} | {'chi2/dof':>9s} {'p':>6s} | result"
    )
    for r in results:
        print(
            f"{r['N']:3d} {r['a']:5.2f} {r['m']:5.2f} {r['w']:5.2f} {r['acc']:5.3f} | "
            f"{r['m_rec']:7.4f} {r['m_err']:7.4f} {r['z_m']:+5.1f}σ | "
            f"{r['w_rec']:7.4f} {r['w_err']:7.4f} {r['z_w']:+5.1f}σ | "
            f"{r['gap'] / r['w']:6.4f} ({r['gap_lattice'] / r['w']:.4f}) | "
            f"{r['chi2']:5.1f}/{r['dof']:<3d} {r['p']:6.3f} | "
            f"{'PASS' if r['passed'] else 'FAIL'}"
        )
    for r in results:
        if r["block_over_tau"] < MIN_BLOCK:
            print(
                f"  warning: N={r['N']} a={r['a']} m={r['m']} w={r['w']}: blocks "
                f"are only {r['block_over_tau']:.0f} tau_int long "
                f"(want {MIN_BLOCK}); increase --sweeps"
            )
    n_fail = sum(not r["passed"] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} runs pass.")
    if len(results) > 1:
        pulls = np.array([[r["z_m"], r["z_w"]] for r in results])
        mean, spread = pulls.mean(axis=0), pulls.std(axis=0, ddof=1)
        print(
            f"pulls: mean m {mean[0]:+.2f}, w {mean[1]:+.2f};  "
            f"spread m {spread[0]:.2f}, w {spread[1]:.2f}  "
            "(honest error bars give mean ~0 and spread ~1)"
        )


def plot_results(results, title=""):
    """
    Plot the correlators, the pulls and the lattice energy gap.

    Left: measured correlator with the fitted exact formula (all j, not just the
    fit range). Middle: pulls (recovered - injected) / error, with the pass
    limit. Right: lattice energy gap / w against a w, with the exact lattice
    curve and the continuum value 1.

    Parameters
    ----------
    results : list of dict
        Outputs of `run_case`, one per case.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    names = [f"N={r['N']}, a={r['a']:g}, m={r['m']:g}, w={r['w']:g}" for r in results]
    for i, (r, name) in enumerate(zip(results, names)):
        color = f"C{i % 10}"
        j = np.arange(len(r["C"]))
        ax[0].errorbar(
            j * r["a"], r["C"], yerr=r["C_err"], fmt="o", ms=3, color=color, label=name
        )
        jj = np.linspace(0, j[-1], 200)
        fit = model_correlator(jj, r["N"], r["a"], r["m_rec"], r["w_rec"])
        ax[0].plot(jj * r["a"], fit, color=color, lw=1)
        ax[1].plot(r["z_m"], i + 0.1, "o", color=color)
        ax[1].plot(r["z_w"], i - 0.1, "s", color=color)
        ax[2].errorbar(
            r["a"] * r["w"],
            r["gap"] / r["w"],
            yerr=r["gap_err"] / r["w"],
            fmt="o",
            color=color,
            capsize=3,
        )
    ax[0].set(
        yscale="log",
        xlabel="time $t = ja$",
        ylabel=r"$C(t)$",
        title="Measured correlator and fitted exact formula",
    )
    ax[0].legend(fontsize=7)

    ax[1].axvspan(-1, 1, color="C2", alpha=0.2)
    ax[1].axvspan(-2, 2, color="C2", alpha=0.1)
    for s in (-Z_MAX, Z_MAX):
        ax[1].axvline(s, color="r", ls="--")
    ax[1].set(
        yticks=range(len(results)),
        yticklabels=names,
        xlim=(-5, 5),
        xlabel="(recovered − injected) / error   [circle: m, square: w]",
        title=f"Pulls (pass limit ±{Z_MAX})",
    )

    aw = np.linspace(0.05, 1.2 * max(r["a"] * r["w"] for r in results), 200)
    ax[2].plot(aw, lattice_mu(aw, 1) / aw, "k", lw=1, label="exact lattice")
    ax[2].axhline(1, color="C3", ls="--", lw=1, label="continuum")
    ax[2].set(
        xlabel=r"$a\omega$",
        ylabel=r"$(E_1 - E_0)\,/\,\omega$",
        title="Lattice energy gap",
    )
    ax[2].legend(fontsize=8)
    fig.suptitle(title)
    return fig


def main(argv=None):
    """
    Run the injection test from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default is ``sys.argv[1:]``.
    """
    p = argparse.ArgumentParser(
        prog="sho-mcmc-inject",
        description="Injection-recovery test (validation plan, Test 10).",
    )
    p.add_argument("--sweeps", type=int, default=100_000, help="sweeps per case")
    p.add_argument("--burn", type=int, default=5_000, help="burn-in sweeps")
    p.add_argument("--delta", type=float, default=1.0, help="proposal half-width")
    p.add_argument("--blocks", type=int, default=100, help="number of blocks")
    p.add_argument("--seed", type=int, default=2026, help="base random seed")
    p.add_argument(
        "--repeats", type=int, default=1, help="repeat every case with new seeds"
    )
    p.add_argument("--out", default=None, help="prefix for the saved figure and json")
    p.add_argument("--no-show", action="store_true", help="do not open the figure")
    args = p.parse_args(argv)

    results = []
    for rep in range(args.repeats):
        for i, (N, a, m, w) in enumerate(CASES):
            seed = args.seed + 1000 * rep + i  # run i of repeat rep
            r = run_case(
                N, a, m, w, args.sweeps, args.burn, args.delta, args.blocks, seed
            )
            results.append(r)
    print_table(results)

    # negative control: sample with w = 1.1 but compare with w = 1; must be caught
    settings = (args.sweeps, args.burn, args.delta, args.blocks)
    bad = run_case(32, 0.5, 1.0, 1.1, *settings, args.seed + 999)
    z_bad = (bad["w_rec"] - 1.0) / bad["w_err"]
    caught = bool(abs(z_bad) > Z_MAX)
    verdict = "caught, as it should be" if caught else "NOT caught: test too weak"
    print(
        "negative control: sampled with w = 1.1, compared with w = 1.0: "
        f"pull {z_bad:+.1f}σ -> {verdict}"
    )
    negative = {"w_sampled": 1.1, "w_compared": 1.0, "pull": z_bad, "caught": caught}

    title = f"Injection test, seed {args.seed}"
    fig = plot_results(results[: len(CASES)], title=title)
    if args.out:
        fig.savefig(f"{args.out}.png", dpi=130)
        out = {
            "settings": vars(args),
            "cases": CASES,
            "results": results,
            "negative_control": negative,
        }
        with open(f"{args.out}.json", "w") as f:
            json.dump(out, f, indent=1)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
