"""Summary of a run and diagnostic figures."""
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import binning_error
from .lattice import action, exact_x2


def summary(chain, acc_rate, N, a, m, w, burn, filename=None):
    """
    Print <x^2> with a binning error bar next to the exact values.

    Every sweep after `burn` is used; no thinning is needed because binning
    accounts for the correlations. Also plots the error against block size;
    the error bar is trustworthy once that curve reaches a plateau.

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
    filename : str, optional
        If given, the figure is saved there.

    Returns
    -------
    mc : float
        Monte Carlo estimate of <x^2>.
    err : float
        Its error bar.
    lat : float
        Exact lattice value.
    cont : float
        Continuum ground-state value 1 / (2 m w).
    """
    series = np.mean(chain[burn:] ** 2, axis=1)  # one <x^2> per sweep
    mc = series.mean()
    sizes, errs, errs_err = binning_error(series)
    err = errs[-1]
    tau = 0.5 * (err / errs[0]) ** 2  # integrated autocorrelation time (sweeps)
    lat, cont = exact_x2(N, a, m, w), 1 / (2 * m * w)
    print(
        f"N={N} a={a} m={m} w={w}  acc={acc_rate:.3f}  tau_int~{tau:.1f} sweeps\n"
        f"  <x^2>: MC={mc:.4f} ± {err:.4f}  lattice={lat:.4f} "
        f"({(mc - lat) / err:+.1f} sigma)  continuum={cont:.4f}"
    )

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.errorbar(sizes, errs, yerr=errs_err, marker="o", capsize=3)
    ax.set(
        xscale="log",
        xlabel="block size (sweeps)",
        ylabel=r"error of $\langle x^2\rangle$",
        title=f"Binning analysis (N={N}, a={a}, w={w})",
    )
    if filename:
        fig.savefig(filename, dpi=130)
    return mc, err, lat, cont


def plot_diagnostics(chain, a, m, w, burn, thin, t0=None, filename=None):
    """
    Walker trace, histogram against |psi_0|^2, four path samples and the action.

    Parameters
    ----------
    chain : ndarray, shape (n_sweeps + 1, N)
        Output of `metropolis`.
    a, m, w : float
        Lattice spacing, mass and frequency.
    burn : int
        Number of initial sweeps to discard (marked on the trace).
    thin : int
        Every `thin`-th sweep is used for the histogram.
    t0 : int, optional
        Time slice shown in the trace; default N // 2.
    filename : str, optional
        If given, the figure is saved there.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    n_sweeps, N = chain.shape[0] - 1, chain.shape[1]
    t0 = N // 2 if t0 is None else t0
    # every site is a draw from the thermal density, close to |psi_0|^2 at large N a
    samples = chain[burn::thin].ravel()
    x2_lat = exact_x2(N, a, m, w)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax[0, 0].plot(chain[:, t0], lw=0.5)
    ax[0, 0].axvline(burn, color="k", ls="--", label="burn-in")
    ax[0, 0].set(title="Walker evolution", xlabel="Sweep", ylabel=f"$x_{{{t0}}}$")
    ax[0, 0].legend()

    ax[0, 1].hist(samples, bins=100, density=True, alpha=0.6, label="MCMC")
    r = 6 * np.sqrt(x2_lat)
    xs = np.linspace(-r, r, 400)
    gauss = np.exp(-(xs**2) / (2 * x2_lat)) / np.sqrt(2 * np.pi * x2_lat)
    psi0 = np.sqrt(m * w / np.pi) * np.exp(-m * w * xs**2)
    ax[0, 1].plot(xs, gauss, label="exact lattice")
    ax[0, 1].plot(xs, psi0, "--", label=r"continuum $|\psi_0|^2$")
    ax[0, 1].set(title="Samples vs ground-state density", xlabel="x", ylabel="density")
    ax[0, 1].legend()

    for s in np.linspace(burn, n_sweeps, 4, dtype=int):
        ax[1, 0].plot(chain[s], lw=1, label=f"sweep {s}")
    ax[1, 0].set(title="Four path samples", xlabel="time slice $t$", ylabel="$x_t$")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(action(chain, a, m, w), lw=0.5)
    ax[1, 1].axhline(N / 2, color="r", ls="--", label=r"$\langle S\rangle = N/2$")
    ax[1, 1].set(title="Action vs sweep", xlabel="Sweep", ylabel="$S[x]$", yscale="log")
    ax[1, 1].legend()

    if filename:
        fig.savefig(filename, dpi=130)
    return fig
