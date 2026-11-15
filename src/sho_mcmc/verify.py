"""
Physics checks for the lattice oscillator against exact results.

Every Monte Carlo value gets a binning error bar and is compared with the exact
lattice value and with the continuum value at beta = N a:

* <x> = 0 and <p> = 0 by symmetry (<p> is exactly 0 for every periodic path)
* <V> = m w^2 <x^2> / 2
* <E> from the primitive estimator and from the virial estimator (<T> = <V>)
* the virial theorem, <T_prim> against <V>
* the autocorrelation function of x^2 and its integrated autocorrelation time
"""

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import autocorrelation, binning_error
from .lattice import exact_values

LABELS = {
    "x": r"$\langle x\rangle$",
    "V": r"$\langle V\rangle$",
    "T_prim": r"$\langle T\rangle_{prim}$",
    "E_prim": r"$\langle E\rangle_{prim}$",
    "E_vir": r"$\langle E\rangle_{vir}$",
}


def measure(chain, a, m, w):
    """
    One value per sweep of each observable, averaged over time slices.

    Parameters
    ----------
    chain : ndarray, shape (n, N)
        Paths after burn-in.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    dict of ndarray
        Per-sweep values of "x", "p", "V", "T_prim", "E_prim", "E_vir" and "x2".
    """
    dx = np.roll(chain, -1, axis=1) - chain
    x2 = np.mean(chain**2, axis=1)
    V = 0.5 * m * w**2 * x2
    T_prim = 1 / (2 * a) - m * np.mean(dx**2, axis=1) / (2 * a**2)
    return {
        "x": np.mean(chain, axis=1),
        "p": m * np.mean(dx, axis=1) / a,  # telescopes to 0 on a periodic path
        "V": V,
        "T_prim": T_prim,
        "E_prim": T_prim + V,
        "E_vir": 2 * V,  # virial: <T> = <x V'(x)> / 2 = <V>
        "x2": x2,
    }


def plot_comparison(results, lattice, continuum, title=""):
    """
    Compare Monte Carlo and continuum values with the exact lattice values.

    Left: MC - lattice with error bars (should sit on 0), and continuum - lattice
    (the discretization error, expected to be nonzero at finite a). Right: pull
    (MC - lattice) / err with 1 and 2 sigma bands.

    Parameters
    ----------
    results : dict
        (mean, err) for each observable in `LABELS`.
    lattice, continuum : dict
        Exact lattice and continuum values.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    keys = list(LABELS)
    y = np.arange(len(keys))
    diff = np.array([results[k][0] - lattice[k] for k in keys])
    err = np.array([results[k][1] for k in keys])
    gap = np.array([continuum[k] - lattice[k] for k in keys])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    ax[0].axvline(0, color="k", lw=1, label="exact lattice")
    ax[0].errorbar(diff, y, xerr=err, fmt="o", capsize=4, label="MC ± err")
    ax[0].plot(gap, y, "s", mfc="none", color="C3", label="continuum")
    ax[0].set(
        yticks=y,
        yticklabels=[LABELS[k] for k in keys],
        xlabel="value − exact lattice",
        title="MC and continuum relative to exact lattice",
    )
    ax[0].legend(fontsize=8)

    pull = diff / err
    ax[1].axvspan(-2, 2, color="C2", alpha=0.12, label=r"$\pm 2\sigma$")
    ax[1].axvspan(-1, 1, color="C2", alpha=0.25, label=r"$\pm 1\sigma$")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].plot(pull, y, "o")
    lim = max(3.5, np.abs(pull).max() + 0.5)
    ax[1].set(
        xlim=(-lim, lim),
        xlabel=r"(MC − lattice) / err  [$\sigma$]",
        title="Pull",
    )
    ax[1].legend(fontsize=8)
    fig.suptitle(title)
    return fig


def plot_distributions(chain, obs, lattice, m, w, title=""):
    """
    Histograms of the per-sweep values after burn-in.

    Panels 1-5 show the per-sweep values of each observable; their width is the
    sweep-to-sweep spread, not the error of the mean. Panel 6 pools every x_t
    and compares with the exact lattice Gaussian and the continuum |psi_0|^2.

    Parameters
    ----------
    chain : ndarray, shape (n, N)
        Paths after burn-in.
    obs : dict of ndarray
        Output of `measure`.
    lattice : dict
        Exact lattice values.
    m, w : float
        Mass and frequency.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axi, k in zip(ax.flat, LABELS):
        axi.hist(obs[k], bins=60, density=True, alpha=0.6)
        axi.axvline(lattice[k], color="k", lw=1.5, label="exact lattice")
        axi.axvline(obs[k].mean(), color="C1", ls="--", lw=1.5, label="MC mean")
        axi.set(title=LABELS[k] + " per sweep", xlabel="value", ylabel="density")
        axi.legend(fontsize=7)

    x2 = 2 * lattice["V"] / (m * w**2)  # exact lattice <x^2>
    xs = np.linspace(-5 * np.sqrt(x2), 5 * np.sqrt(x2), 400)
    gauss = np.exp(-(xs**2) / (2 * x2)) / np.sqrt(2 * np.pi * x2)
    psi0 = np.sqrt(m * w / np.pi) * np.exp(-m * w * xs**2)
    ax[1, 2].hist(
        chain.ravel(), bins=100, density=True, alpha=0.6, label="MC, all $x_t$"
    )
    ax[1, 2].plot(xs, gauss, "k", label="exact lattice")
    ax[1, 2].plot(xs, psi0, "C3--", label=r"continuum $|\psi_0|^2$")
    ax[1, 2].set(title="Position distribution", xlabel="x", ylabel="density")
    ax[1, 2].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def verify(chain, acc_rate, N, a, m, w, burn, save=None):
    """
    Print the table of checks and draw the comparison figures.

    Parameters
    ----------
    chain : ndarray, shape (n_sweeps + 1, N)
        Output of `metropolis`.
    acc_rate : float
        Acceptance rate.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    burn : int
        Number of initial sweeps to discard.
    save : str, optional
        Prefix for saved figures, e.g. "run1" gives run1_comparison.png,
        run1_distributions.png and run1_autocorr.png. None only draws them.

    Returns
    -------
    dict
        (mean, err) for each observable.

    Notes
    -----
    Table columns: MC is the simulation's estimate; ± err is its 1 sigma error
    bar from binning; lattice is the exact lattice value; dev is
    (MC - lattice) / err, where anything within ±2 sigma is normal; continuum is
    the exact value at temperature 1 / (N a); plateau says whether the binning
    curve levelled off, i.e. whether the error bar is reliable.
    """
    obs = measure(chain[burn:], a, m, w)
    lattice, continuum = exact_values(N, a, m, w)
    labels = {
        "x": "<x>",
        "p": "<p>",
        "V": "<V>",
        "T_prim": "<T> primitive",
        "E_prim": "<E> primitive",
        "E_vir": "<E> virial",
    }

    print(
        f"N={N} a={a} m={m} w={w} beta={N * a:g}  acc={acc_rate:.3f}  "
        f"sweeps used={len(chain) - 1 - burn}"
    )
    print(
        f"{'observable':15s} {'MC':>9s} {'± err':>8s} {'lattice':>9s} {'dev':>7s} "
        f"{'continuum':>10s}  plateau?"
    )
    results = {}
    for k, name in labels.items():
        mean = obs[k].mean()
        sizes, errs, errs_err = binning_error(obs[k])
        err = errs[-1]
        # plateau check: last two block sizes agree within their errors
        plateau = abs(errs[-1] - errs[-2]) < np.hypot(errs_err[-1], errs_err[-2])
        if err < 1e-12:  # e.g. <p>: zero for every path
            dev, flag = "exact 0", "trivial"
        else:
            dev = f"{(mean - lattice[k]) / err:+.1f}σ"
            flag = "yes" if plateau else "NO"
        print(
            f"{name:15s} {mean:9.4f} {err:8.4f} {lattice[k]:9.4f} {dev:>7s} "
            f"{continuum[k]:10.4f}  {flag}"
        )
        results[k] = (mean, err)

    # virial theorem: <T> = <V> (exactly in the continuum, O(a^2) on the lattice)
    diff = obs["T_prim"] - obs["V"]
    d_err = binning_error(diff)[1][-1]
    exact_diff = lattice["T_prim"] - lattice["V"]
    print(
        f"\nvirial check <T_prim> - <V> = {diff.mean():+.4f} ± {d_err:.4f}"
        f"   (exact lattice {exact_diff:+.4f}, continuum 0)"
    )

    # autocorrelation of x^2
    rho, tau, W = autocorrelation(obs["x2"])
    n = len(obs["x2"])
    err_x2 = binning_error(obs["x2"])[1][-1]
    tau_bin = 0.5 * (err_x2 / obs["x2"].std(ddof=1) * np.sqrt(n)) ** 2
    n_eff = n / (2 * tau)
    print(
        f"x^2 autocorrelation: tau_int = {tau:.1f} sweeps (window {W}), "
        f"binning gives {tau_bin:.1f};  ~{n_eff:.0f} independent samples"
    )
    if n_eff < 1000:
        print("  warning: fewer than ~1000 independent samples; increase --sweeps")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    lag = np.arange(min(len(rho), 5 * W))
    ax[0].plot(lag, rho[lag])
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axvline(W, color="r", ls="--", label=f"window W={W}")
    ax[0].set(
        title=r"Autocorrelation of $x^2$",
        xlabel="lag (sweeps)",
        ylabel=r"$\rho(t)$",
    )
    ax[0].legend()
    taus = 0.5 + np.cumsum(rho[1 : len(lag)])
    ax[1].plot(lag[1:], taus)
    ax[1].axhline(tau, color="r", ls="--", label=rf"$\tau_{{int}}$ = {tau:.1f}")
    ax[1].set(
        title="Integrated autocorrelation time vs window",
        xlabel="window (sweeps)",
        ylabel=r"$\tau_{int}$",
    )
    ax[1].legend()

    title = f"N={N}, a={a}, m={m}, w={w}"
    figs = {
        "autocorr": fig,
        "comparison": plot_comparison(results, lattice, continuum, title),
        "distributions": plot_distributions(chain[burn:], obs, lattice, m, w, title),
    }
    if save:
        for name, f in figs.items():
            f.savefig(f"{save}_{name}.png", dpi=130)
    return results
