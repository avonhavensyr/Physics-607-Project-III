"""
Physics checks for the lattice harmonic oscillator, against exact results.

Kevin's verification code. The hand-written Metropolis
(`sho_mcmc.metropolis.metropolis`) is the sampler under test. emcee
(`sho_mcmc.ensemble.emcee_sampler`) is only a reference to compare it against:
it samples the same target, and every table and figure shows hand-written vs
lattice, emcee vs lattice, and emcee vs hand-written. Run it with the
``sho-mcmc`` command; add ``--emcee`` to include emcee.

Checks (each MC value has an error bar, the larger of two estimates;
Metropolis: binning plateau vs autocorrelation time; emcee: spread between
walkers vs binning/autocorrelation of the walker average over time):

* <x> = 0 and <p> = 0 by symmetry (<p> is identically 0 on a periodic lattice)
* <V> = m w^2 <x^2> / 2, against the exact lattice and the continuum at beta = N a
* <E> from the primitive and the virial estimator (<T> = <V> for the oscillator)
* the virial theorem, <T_primitive> against <V>
* the autocorrelation function of x^2 and its integrated autocorrelation time
* hand-written Metropolis against emcee: (emcee - MH) / combined error

Symmetry checks (separate table and figure; all exactly 0 by symmetry):

* parity x -> -x: <x> = 0 and <x^3> = 0 (<p> = 0 is shown but trivial)
* time translation t -> t + k: every slice has the same <x_t^2>; summary
  numbers are <x^2> of the first half minus the second half of the slices, and
  of the even minus the odd slices (which would catch a checkerboard bug)
* time reversal t -> -t: forward minus backward correlator from the same
  origins, <x_t x_{t+j}> - <x_t x_{t-j}> for t = 0..N/2-1. Averaged over all
  origins this is 0 for every path, so it would test nothing; a fixed set of
  origins can fail.
* the origin-averaged correlator C(j) against the exact lattice correlator
"""

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import binning_error
from .lattice import exact_correlator


# ---------------- exact results ----------------
def exact_values(N, a, m, w):
    """
    Exact lattice and continuum values of every reported quantity.

    The continuum values are for the oscillator at temperature 1/beta, with
    beta = N a. The symmetry checks are exactly 0 in both.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    lattice, continuum : dict
        Exact values, keyed like the output of `measure`.
    """
    C = exact_correlator(N, a, m, w)
    x2, dx2 = C[0], 2 * (C[0] - C[1])  # <x^2>, <(x_{t+1}-x_t)^2>
    V = 0.5 * m * w**2 * x2
    T_prim = 1 / (2 * a) - m * dx2 / (2 * a**2)
    coth = 1 / np.tanh(N * a * w / 2)
    lattice = {
        "x": 0.0,
        "p": 0.0,
        "V": V,
        "T_prim": T_prim,
        "E_prim": T_prim + V,
        "E_vir": 2 * V,
    }
    continuum = {
        "x": 0.0,
        "p": 0.0,
        "V": w / 4 * coth,
        "T_prim": w / 4 * coth,
        "E_prim": w / 2 * coth,
        "E_vir": w / 2 * coth,
    }
    for k in ("x3", "tt_half", "tt_parity", "tr"):
        lattice[k] = continuum[k] = 0.0
    return lattice, continuum


# ---------------- measurements ----------------
# A chain is either (n_sweeps, N) from metropolis, or (n_rows, n_walkers, N) from emcee.
def measure(chain, a, m, w):
    """
    Every observable for every stored path, averaged over time slices.

    Parameters
    ----------
    chain : ndarray, shape (n_sweeps, N) or (n_rows, n_walkers, N)
        Paths from `metropolis` or from `emcee_sampler`.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    dict of ndarray
        Shape (n_sweeps,) for Metropolis or (n_rows, n_walkers) for emcee;
        "D_j" has one more axis, of length N/2 - 1.
    """
    N = chain.shape[-1]
    dx = np.roll(chain, -1, axis=-1) - chain
    sq = chain**2
    x2 = np.mean(sq, axis=-1)
    V = 0.5 * m * w**2 * x2
    T_prim = 1 / (2 * a) - m * np.mean(dx**2, axis=-1) / (2 * a**2)
    return {
        "x": np.mean(chain, axis=-1),
        "p": m * np.mean(dx, axis=-1) / a,  # telescopes to 0 on a periodic path
        "V": V,
        "T_prim": T_prim,
        "E_prim": T_prim + V,
        "E_vir": 2 * V,  # virial: <T> = 1/2 <x V'(x)> = <V>
        "x2": x2,
        # symmetry checks (exactly 0 by symmetry)
        "x3": np.mean(chain**3, axis=-1),  # parity
        "tt_half": sq[..., : N // 2].mean(-1)
        - sq[..., N // 2 :].mean(-1),  # translation
        "tt_parity": sq[..., 0::2].mean(-1) - sq[..., 1::2].mean(-1),  # translation
        "D_j": (D := reversal_differences(chain)),
        "tr": D.mean(-1),  # reversal
    }


def reversal_differences(chain):
    """
    Time-reversal differences over a fixed set of origins.

    D(j) = <x_t x_{t+j}> - <x_t x_{t-j}>, averaged over the origins
    t = 0..N/2-1, for j = 1..N/2-1.

    Parameters
    ----------
    chain : ndarray, shape (..., N)
        Paths.

    Returns
    -------
    ndarray, shape (..., N/2 - 1)
        One value per stored path and separation j.
    """
    N = chain.shape[-1]
    T = np.arange(N // 2)
    x0 = chain[..., T]
    D = [
        np.mean(x0 * chain[..., j : j + N // 2], -1)
        - np.mean(x0 * chain[..., (T - j) % N], -1)
        for j in range(1, N // 2)
    ]
    return np.stack(D, axis=-1)


def symmetry_profiles(chain, D):
    """
    Per-slice <x_t^2>, the origin-averaged correlator C(j), and D(j).

    Parameters
    ----------
    chain : ndarray, shape (..., N)
        Paths after burn-in.
    D : ndarray
        Output of `reversal_differences`.

    Returns
    -------
    dict
        "x2_t", "C" and "D", each a pair (values, errors), with the errors from
        `mean_and_error` (binning for Metropolis, walker spread for emcee).
    """
    N = chain.shape[-1]
    f = np.fft.rfft(chain, axis=-1)
    C = np.fft.irfft(f * np.conj(f), n=N, axis=-1) / N  # (1/N) sum_t x_t x_{t+j}
    sq = chain**2

    def per_index(arr):
        """Mean and error for every index of the last axis."""
        out = np.array([mean_and_error(arr[..., i])[:2] for i in range(arr.shape[-1])])
        return out[:, 0], out[:, 1]

    return {
        "x2_t": per_index(sq),
        "C": per_index(C[..., : N // 2 + 1]),
        "D": per_index(D),
    }


def _time_series_error(y):
    """
    Two independent error estimates for the mean of a correlated series.

    binning: the average of the block errors over the plateau region (block
    size >= 10 tau, with >= 32 blocks), which is far less noisy than the last
    block alone. tau: sqrt(2 tau_int var / n), from the autocorrelation
    function.

    Parameters
    ----------
    y : ndarray
        One-dimensional series.

    Returns
    -------
    err_bin : float or None
        Binning error, or None if there is no plateau region.
    err_tau : float
        Error from the integrated autocorrelation time.
    tau : float
        Integrated autocorrelation time.
    """
    _, tau, _ = autocorrelation(y)
    err_tau = np.sqrt(2 * tau * y.var(ddof=1) / len(y))
    sizes, errs, _ = binning_error(y)
    plateau = errs[sizes >= 10 * tau]
    return (plateau.mean() if len(plateau) else None), err_tau, tau


def mean_and_error(series):
    """
    Mean and error bar; the error is the larger of the available estimates.

    Metropolis (1-D series): binning (plateau average) and autocorrelation
    estimates. The flag is "yes" when the binning curve reaches a plateau and
    the two estimates agree within 30%, and "NO" when there is no plateau
    region (a run shorter than about 320 tau) or the estimates disagree; then
    the run should be longer.

    emcee (rows x walkers): the spread of the walkers' time averages divided by
    sqrt(n_walkers), and binning and autocorrelation of the walker-averaged
    series over time. The time-based estimate also captures correlations
    between walkers, which the spread misses. The flag says which estimate set
    the error: "walkers" or "time".

    Parameters
    ----------
    series : ndarray, shape (n,) or (n_rows, n_walkers)
        Values of one observable.

    Returns
    -------
    mean : float
        The mean.
    err : float
        The error bar.
    flag : str
        See above.
    """
    if series.ndim == 1:
        if series.var() < 1e-30:  # constant series, e.g. <p>
            return series.mean(), 0.0, "yes"
        err_bin, err_tau, _ = _time_series_error(series)
        if err_bin is None:
            return series.mean(), err_tau, "NO"
        agree = abs(err_bin - err_tau) < 0.3 * max(err_bin, err_tau)
        return series.mean(), max(err_bin, err_tau), "yes" if agree else "NO"

    walker_means = series.mean(axis=0)
    err_walk = walker_means.std(ddof=1) / np.sqrt(len(walker_means))
    y = series.mean(axis=1)  # ensemble average per stored step
    if y.var() < 1e-30:
        return series.mean(), 0.0, "walkers"
    err_bin, err_tau, _ = _time_series_error(y)
    err_time = max(err_tau, err_bin or 0.0)
    return (
        series.mean(),
        max(err_walk, err_time),
        "walkers" if err_walk >= err_time else "time",
    )


def autocorrelation(series, c=6.0):
    """
    Normalized autocorrelation and integrated autocorrelation time.

    tau_int = 1/2 + sum rho(t) is summed up to the first window W with
    W >= c tau_int(W) (Sokal's automatic window). For emcee (rows x walkers)
    the autocorrelation is averaged over walkers. Same convention as the
    binning estimate: err^2 = 2 tau_int var / n.

    Parameters
    ----------
    series : ndarray, shape (n,) or (n, n_walkers)
        Values of one observable.
    c : float, optional
        Window constant.

    Returns
    -------
    rho : ndarray
        Normalized autocorrelation function.
    tau : float
        Integrated autocorrelation time, in stored rows.
    W : int
        Window used for the sum.
    """
    y = np.asarray(series, dtype=float)
    y = (y - y.mean(axis=0)).reshape(len(y), -1)  # (n, walkers); walkers = 1 for MH
    n = len(y)
    f = np.fft.rfft(y, 2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(axis=1)
    rho = acf / acf[0]
    taus = 0.5 + np.cumsum(rho[1:])
    W = next((t for t in range(1, n) if t >= c * taus[t - 1]), n - 1)
    return rho, taus[W - 1], W


# ---------------- analysis of one sampler ----------------
NAMES = {
    "V": "<V>",
    "T_prim": "<T> primitive",
    "E_prim": "<E> primitive",
    "E_vir": "<E> virial",
}
SYM_NAMES = {
    "x": "<x>",
    "x3": "<x^3>",
    "p": "<p>",
    "tt_half": "x^2 half1-half2",
    "tt_parity": "x^2 even-odd",
    "tr": "corr fwd-bwd",
}
LABELS = {
    "V": r"$\langle V\rangle$",
    "T_prim": r"$\langle T\rangle_{prim}$",
    "E_prim": r"$\langle E\rangle_{prim}$",
    "E_vir": r"$\langle E\rangle_{vir}$",
}
SYM_LABELS = {
    "x": r"$\langle x\rangle$",
    "x3": r"$\langle x^3\rangle$",
    "tt_half": r"$x^2$: 1st − 2nd half",
    "tt_parity": r"$x^2$: even − odd",
    "tr": "corr: fwd − bwd",
}
DIST_LABELS = {"x": SYM_LABELS["x"], **LABELS}


def analyze(chain, acc_rate, burn, thin, a, m, w):
    """
    All measurements for one sampler.

    Parameters
    ----------
    chain : ndarray
        Metropolis chain (n_sweeps + 1, N) or emcee chain (n_rows, n_walkers, N).
    acc_rate : float
        Acceptance rate.
    burn : int
        Rows to discard (sweeps for Metropolis, stored rows for emcee).
    thin : int
        Steps between stored rows (1 for Metropolis); used to report times in
        steps.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    dict
        Measurements, (mean, err, flag) for every quantity, autocorrelation
        data and symmetry profiles.
    """
    kept = chain[burn:]
    obs = measure(kept, a, m, w)
    is_mh = kept.ndim == 2
    rho, tau, W = autocorrelation(obs["x2"])
    results = {
        k: mean_and_error(obs[k]) for k in {**NAMES, **SYM_NAMES}
    }  # (mean, err, flag)
    vir = obs["T_prim"] - obs["V"]
    return {
        "kept": kept,
        "obs": obs,
        "results": results,
        "acc": acc_rate,
        "thin": thin,
        "is_mh": is_mh,
        "unit": "sweeps" if is_mh else "steps",
        "walkers": 1 if is_mh else kept.shape[1],
        "rho": rho,
        "tau": tau,
        "W": W,
        "n_eff": obs["x2"].size / (2 * tau),
        "run_in_tau": len(kept) / tau,
        "virial": (vir.mean(), mean_and_error(vir)[1]),
        "profiles": symmetry_profiles(kept, obs["D_j"]),
    }


# ---------------- combined report ----------------
def print_report(runs, N, a, m, w):
    """
    Print the tables of results for all samplers.

    Parameters
    ----------
    runs : dict
        Output of `analyze` for each sampler, keyed by sampler name.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    """
    lattice, continuum = exact_values(N, a, m, w)
    names = list(runs)
    print(f"N={N} a={a} m={m} w={w} beta={N * a:g}")
    for nm, r in runs.items():
        used = f"{len(r['kept']) * r['thin']} {r['unit']}"
        if not r["is_mh"]:
            used += f" x {r['walkers']} walkers"
        err_from = (
            "max(binning, autocorrelation)"
            if r["is_mh"]
            else "max(walker spread, time binning/autocorrelation)"
        )
        print(
            f"  {nm:11s} acc={r['acc']:.3f}  used={used}  "
            f"tau_int(x^2)={r['tau'] * r['thin']:.1f} {r['unit']}  "
            f"~{r['n_eff']:.0f} independent samples  errors: {err_from}"
        )

    no_plateau = _print_table("Physics", NAMES, runs, lattice, continuum)
    no_plateau |= _print_table(
        "Symmetry checks (all exactly 0)", SYM_NAMES, runs, lattice, continuum
    )
    names = list(runs)
    print(
        "  σ columns: (MC - exact) / err"
        + (
            f"; last column: ({names[1]} - {names[0]}) / combined err"
            if len(names) == 2
            else ""
        )
    )
    if no_plateau:
        print(
            "  ! binning and autocorrelation errors disagree, or no plateau: "
            "treat that error bar with caution; run longer"
        )

    lattice_C = exact_correlator(N, a, m, w)
    for nm, r in runs.items():
        m2, e2 = r["profiles"]["x2_t"]
        mc, ec = r["profiles"]["C"]
        md, ed = r["profiles"]["D"]
        out_t = np.sum(np.abs((m2 - lattice_C[0]) / e2) > 2)
        out_c = np.sum(np.abs((mc - lattice_C[: len(mc)]) / ec) > 2)
        out_d = np.sum(np.abs(md / ed) > 2)
        print(
            f"  {nm:11s} beyond 2σ: <x_t^2> {out_t}/{N} slices, C(j) {out_c}/{len(mc)}, "
            f"fwd-bwd {out_d}/{len(md)}  (expect ~5%; neighbours are correlated)"
        )

    print(
        f"\nvirial check <T_prim> - <V>   (exact lattice "
        f"{lattice['T_prim'] - lattice['V']:+.4f}, continuum 0)"
    )
    for nm, r in runs.items():
        print(f"  {nm:11s} {r['virial'][0]:+.4f} ± {r['virial'][1]:.4f}")

    for nm, r in runs.items():
        if r["n_eff"] < 1000:
            print(f"  warning ({nm}): fewer than ~1000 independent samples; run longer")
        if not r["is_mh"] and r["run_in_tau"] < 50:
            print(
                f"  warning ({nm}): run is only {r['run_in_tau']:.0f} tau long per walker "
                "(want >= 50); walker-spread errors may be unreliable"
            )


def _print_table(title, names_dict, runs, lattice, continuum):
    """
    Print one table: each sampler's value, error and pull, and the exact values.

    Parameters
    ----------
    title : str
        Table title.
    names_dict : dict
        Observable key -> printed name.
    runs : dict
        Output of `analyze` for each sampler.
    lattice, continuum : dict
        Exact values.

    Returns
    -------
    bool
        True if any error bar is flagged as unreliable.
    """
    names = list(runs)
    cell, lw = 26, 17
    head = f"\n{title}\n{'observable':{lw}s}" + "".join(
        f"{nm:>{cell}s}" for nm in names
    )
    head += f"{'lattice':>10s}{'continuum':>11s}"
    if len(names) == 2:
        head += f"{names[1][:5]}-{names[0][:4]}".rjust(14)
    print(head)
    no_plateau = False
    for k, label in names_dict.items():
        row = f"{label:{lw}s}"
        for nm in names:
            mean, err, flag = runs[nm]["results"][k]
            if err < 1e-12:
                txt = f"{mean:+.4f} (trivial)"
            else:
                txt = f"{mean:+.4f}±{err:.4f} {(mean - lattice[k]) / err:+.1f}σ"
                if flag == "NO":
                    txt += "!"
                    no_plateau = True
            row += f"{txt:>{cell}s}"
        row += f"{lattice[k]:10.4f}{continuum[k]:11.4f}"
        if len(names) == 2:
            (ma, ea, _), (mb, eb, _) = (runs[nm]["results"][k] for nm in names)
            de = np.hypot(ea, eb)
            row += f"{'—' if de < 1e-12 else f'{(mb - ma) / de:+.1f}σ':>14s}"
        print(row)
    return no_plateau


# ---------------- combined figures ----------------
def _pull_axes(ax, title, xlabel):
    """
    Draw the 1 and 2 sigma bands of a pull plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    title, xlabel : str
        Title and x label.
    """
    ax.axvspan(-2, 2, color="0.5", alpha=0.12, label=r"$\pm 2\sigma$")
    ax.axvspan(-1, 1, color="0.5", alpha=0.25, label=r"$\pm 1\sigma$")
    ax.axvline(0, color="k", lw=1)
    ax.set(title=title, xlabel=xlabel)


def plot_comparison(runs, N, a, m, w, title="", labels=None, show_continuum=True):
    """
    Differences from the exact lattice values, and pulls, mirroring the table.

    With two samplers (hand-written first, emcee second): hand-written minus
    exact lattice, emcee minus exact lattice, emcee minus hand-written with the
    combined error (all should sit on 0), and all three as pulls in sigma.
    Continuum minus lattice (red squares) shows the discretization error at
    finite a. With one sampler: its difference from lattice, and its pull.

    Parameters
    ----------
    runs : dict
        Output of `analyze` for each sampler.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    title : str, optional
        Figure title.
    labels : dict, optional
        Observables to show and their labels; default `LABELS`.
    show_continuum : bool, optional
        Whether to show continuum minus lattice.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    lattice, continuum = exact_values(N, a, m, w)
    labels = LABELS if labels is None else labels
    keys, names = list(labels), list(runs)
    y = np.arange(len(keys))
    gap = np.array([continuum[k] - lattice[k] for k in keys])

    def vals(nm):
        """Means and errors of one sampler for the shown observables."""
        res = runs[nm]["results"]
        return (
            np.array([res[k][0] for k in keys]),
            np.array([res[k][1] for k in keys]),
        )

    panels = []  # (title, diff, err, color, show continuum)
    for i, nm in enumerate(names):
        mean, err = vals(nm)
        lat = np.array([lattice[k] for k in keys])
        panels.append(
            (f"{nm} − exact lattice", mean - lat, err, f"C{i}", show_continuum)
        )
    if len(names) == 2:
        (ma, ea), (mb, eb) = vals(names[0]), vals(names[1])
        panels.append(
            (f"{names[1]} − {names[0]}", mb - ma, np.hypot(ea, eb), "C4", False)
        )

    ncols = 2
    nrows = int(np.ceil((len(panels) + 1) / ncols))
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(12, 4 * nrows), constrained_layout=True, squeeze=False
    )
    ax = ax.ravel()
    for axi, (ttl, diff, err, col, show_cont) in zip(ax, panels):
        axi.axvline(0, color="k", lw=1)
        axi.errorbar(
            diff, y, xerr=err, fmt="o", capsize=4, color=col, label="difference ± err"
        )
        if show_cont:
            axi.plot(gap, y, "s", mfc="none", color="C3", label="continuum − lattice")
        axi.set(
            yticks=y,
            yticklabels=[labels[k] for k in keys],
            title=ttl,
            xlabel="difference",
        )
        axi.legend(fontsize=8)

    pax = ax[len(panels)]
    _pull_axes(pax, "Pulls (difference / error)", r"$\sigma$")
    off = np.linspace(0.2, -0.2, len(panels))
    lim = 3.5
    for (ttl, diff, err, col, _), o in zip(panels, off):
        pull = diff / err
        pax.plot(pull, y + o, "D" if col == "C4" else "o", color=col, label=ttl)
        lim = max(lim, np.abs(pull).max() + 0.5)
    pax.set(xlim=(-lim, lim), yticks=y, yticklabels=[labels[k] for k in keys])
    pax.legend(fontsize=8)
    for axi in ax[len(panels) + 1 :]:
        axi.axis("off")
    fig.suptitle(title)
    return fig


def plot_distributions(runs, N, a, m, w, title=""):
    """
    Histograms after burn-in, with all samplers overlaid.

    Panels 1-5: each observable for every stored path (the width is the
    path-to-path spread, not the error of the mean). Panel 6: every x_t pooled,
    against the exact lattice Gaussian and the continuum |psi_0|^2.

    Parameters
    ----------
    runs : dict
        Output of `analyze` for each sampler.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    lattice, _ = exact_values(N, a, m, w)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axi, k in zip(ax.flat, DIST_LABELS):
        for i, (nm, r) in enumerate(runs.items()):
            v = r["obs"][k].ravel()
            axi.hist(
                v,
                bins=60,
                density=True,
                histtype="step",
                lw=1.5,
                color=f"C{i}",
                label=nm,
            )
            axi.axvline(v.mean(), color=f"C{i}", ls="--", lw=1)
        axi.axvline(lattice[k], color="k", lw=1.5, label="exact lattice")
        axi.set(
            title=DIST_LABELS[k] + " per path (dashed: MC mean)",
            xlabel="value",
            ylabel="density",
        )
        axi.legend(fontsize=7)

    x2 = 2 * lattice["V"] / (m * w**2)  # exact lattice <x^2>
    xs = np.linspace(-5 * np.sqrt(x2), 5 * np.sqrt(x2), 400)
    for i, (nm, r) in enumerate(runs.items()):
        ax[1, 2].hist(
            r["kept"].ravel(),
            bins=100,
            density=True,
            histtype="step",
            lw=1.5,
            color=f"C{i}",
            label=nm,
        )
    ax[1, 2].plot(
        xs,
        np.exp(-(xs**2) / (2 * x2)) / np.sqrt(2 * np.pi * x2),
        "k",
        label="exact lattice",
    )
    ax[1, 2].plot(
        xs,
        np.sqrt(m * w / np.pi) * np.exp(-m * w * xs**2),
        "C3--",
        label=r"continuum $|\psi_0|^2$",
    )
    ax[1, 2].set(title="Position distribution", xlabel="x", ylabel="density")
    ax[1, 2].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def plot_autocorrelation(runs, title=""):
    """
    Autocorrelation of x^2 and the running tau_int, one row per sampler.

    Lags are in sweeps (Metropolis) or steps (emcee); the units differ, so the
    rows are separate.

    Parameters
    ----------
    runs : dict
        Output of `analyze` for each sampler.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(
        len(runs),
        2,
        figsize=(11, 3.6 * len(runs)),
        constrained_layout=True,
        squeeze=False,
    )
    for i, (nm, r) in enumerate(runs.items()):
        rho, tau, W, th, unit = r["rho"], r["tau"], r["W"], r["thin"], r["unit"]
        lag = np.arange(min(len(rho), 5 * W))
        ax[i, 0].plot(lag * th, rho[lag], color=f"C{i}")
        ax[i, 0].axhline(0, color="k", lw=0.5)
        ax[i, 0].axvline(W * th, color="r", ls="--", label=f"window W={W * th}")
        ax[i, 0].set(
            title=rf"{nm}: autocorrelation of $x^2$",
            xlabel=f"lag ({unit})",
            ylabel=r"$\rho(t)$",
        )
        ax[i, 0].legend()
        taus = 0.5 + np.cumsum(rho[1 : len(lag)])
        ax[i, 1].plot(lag[1:] * th, taus * th, color=f"C{i}")
        ax[i, 1].axhline(
            tau * th, color="r", ls="--", label=rf"$\tau_{{int}}$ = {tau * th:.1f}"
        )
        ax[i, 1].set(
            title=f"{nm}: integrated autocorrelation time",
            xlabel=f"window ({unit})",
            ylabel=rf"$\tau_{{int}}$ ({unit})",
        )
        ax[i, 1].legend()
    fig.suptitle(title)
    return fig


def plot_symmetries(runs, N, a, m, w, title=""):
    """
    Time translation, the correlator, and time reversal for every sampler.

    Translation: <x_t^2> for every slice t (should be flat at the exact value).
    Correlator: the origin-averaged C(j) against the exact lattice correlator.
    Reversal: <x_t x_{t+j}> - <x_t x_{t-j}> over the origins t = 0..N/2-1
    (should be 0).

    Parameters
    ----------
    runs : dict
        Output of `analyze` for each sampler.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    C_exact = exact_correlator(N, a, m, w)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    for i, (nm, r) in enumerate(runs.items()):
        pr, col = r["profiles"], f"C{i}"
        m2, e2 = pr["x2_t"]
        t = np.arange(len(m2))
        ax[0].fill_between(t, m2 - e2, m2 + e2, color=col, alpha=0.3)
        ax[0].plot(t, m2, color=col, lw=1, label=f"{nm} ± err")
        mc, ec = pr["C"]
        j = np.arange(len(mc))
        ax[1].errorbar(
            j + 0.15 * i, mc, yerr=ec, fmt="o", ms=3, capsize=2, color=col, label=nm
        )
        md, ed = pr["D"]
        jd = np.arange(1, len(md) + 1)
        ax[2].errorbar(
            jd + 0.15 * i, md, yerr=ed, fmt="o", ms=3, capsize=2, color=col, label=nm
        )
    ax[0].axhline(C_exact[0], color="k", lw=1.5, label="exact lattice")
    ax[0].set(
        title=r"Time translation: $\langle x_t^2\rangle$ per slice",
        xlabel="time slice $t$",
        ylabel=r"$\langle x_t^2\rangle$",
    )
    jj = np.arange(N // 2 + 1)
    ax[1].plot(jj, C_exact[jj], "k", lw=1.5, label="exact lattice")
    ax[1].set(
        title=r"Correlator $C(j)=\langle x_t x_{t+j}\rangle$",
        xlabel="separation $j$",
        ylabel="$C(j)$",
    )
    ax[2].axhline(0, color="k", lw=1.5, label="exact (0)")
    ax[2].set(
        title=r"Time reversal: $\langle x_t x_{t+j}\rangle - \langle x_t x_{t-j}\rangle$",
        xlabel="separation $j$",
        ylabel="forward − backward",
    )
    for axi in ax:
        axi.legend(fontsize=8)
    fig.suptitle(title)
    return fig


def verify(runs, N, a, m, w, save=None):
    """
    Analyze every sampler, print the combined tables and draw the figures.

    Parameters
    ----------
    runs : dict
        {name: (chain, acc_rate, burn, thin)} for one or more samplers, e.g.
        {"Metropolis": (chain, acc, burn, 1),
        "emcee": (chain_e, acc_e, burn_rows, thin)}.
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.
    save : str, optional
        Prefix for saved figures, e.g. "run1" gives run1_comparison.png.

    Returns
    -------
    dict
        {name: {observable: (mean, err, flag)}}.

    Notes
    -----
    Table columns: each sampler's estimate with its 1 sigma error bar and
    (MC - exact) / err, where anything within ±2 sigma is normal; lattice is the
    exact lattice value; continuum is the exact value at temperature 1 / (N a).
    A "!" marks an error bar whose binning and autocorrelation estimates
    disagree, or that has no plateau.
    """
    analyses = {
        nm: analyze(c, acc, burn, th, a, m, w)
        for nm, (c, acc, burn, th) in runs.items()
    }
    print_report(analyses, N, a, m, w)
    title = f"{' vs '.join(analyses)}: N={N}, a={a}, m={m}, w={w}"
    figs = {
        "comparison": plot_comparison(analyses, N, a, m, w, title),
        "symmetry_pulls": plot_comparison(
            analyses,
            N,
            a,
            m,
            w,
            "Symmetry checks — " + title,
            SYM_LABELS,
            show_continuum=False,
        ),
        "symmetries": plot_symmetries(analyses, N, a, m, w, title),
        "distributions": plot_distributions(analyses, N, a, m, w, title),
        "autocorr": plot_autocorrelation(analyses, title),
    }
    if save:
        for name, f in figs.items():
            f.savefig(f"{save}_{name}.png", dpi=130)
    return {nm: r["results"] for nm, r in analyses.items()}
