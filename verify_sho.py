"""
Physics verifications for the lattice harmonic oscillator, using sho_mcmc.py.

Set the parameters in the block at the bottom, then run:  python verify_sho.py

The hand-written Metropolis (sho_mcmc.metropolis) is the sampler under test.
emcee's only role is to be a reference to compare it against: it samples the same
target, and every table and figure shows hand-written vs lattice, emcee vs lattice,
and emcee vs hand-written.

Checks (each MC value has an error bar: binning for Metropolis, spread between
walkers for emcee):
  <x>  = 0                                   by symmetry
  <p>  = 0                                   by symmetry (identically 0 on a periodic lattice)
  <V>  = 1/2 m w^2 <x^2>                     vs exact lattice and continuum at beta = N a
  <E>  primitive estimator                   vs exact lattice and continuum
  <E>  virial estimator  (<T> = <V> for SHO) vs continuum
  virial theorem: <T_primitive> vs <V>
  autocorrelation function of x^2 and its integrated autocorrelation time
  hand-written Metropolis vs emcee: (MH - emcee) / combined error
"""
import numpy as np
import matplotlib.pyplot as plt
from sho_mcmc import metropolis, emcee_sampler, coefficients, binning_error
from tqdm import tqdm

# ---------------- exact results ----------------
def exact_correlator(N, a, m, w):
    """Exact lattice C(j) = <x_t x_{t+j}>, j = 0..N-1, periodic lattice."""
    A, B = coefficients(a, m, w)
    theta = 2 * np.pi * np.arange(N) / N
    lam = 2 * A - 2 * B * np.cos(theta)                 # eigenvalues of S = 1/2 x^T K x
    return np.array([np.mean(np.cos(theta * j) / lam) for j in range(N)])


def exact_values(N, a, m, w):
    """Exact lattice and continuum (finite temperature, beta = N a) expectation values."""
    C = exact_correlator(N, a, m, w)
    x2, dx2 = C[0], 2 * (C[0] - C[1])                   # <x^2>, <(x_{t+1}-x_t)^2>
    V = 0.5 * m * w**2 * x2
    T_prim = 1 / (2 * a) - m * dx2 / (2 * a**2)
    coth = 1 / np.tanh(N * a * w / 2)
    lattice = {"x": 0.0, "p": 0.0, "V": V, "T_prim": T_prim, "E_prim": T_prim + V, "E_vir": 2 * V}
    continuum = {"x": 0.0, "p": 0.0, "V": w / 4 * coth, "T_prim": w / 4 * coth,
                 "E_prim": w / 2 * coth, "E_vir": w / 2 * coth}
    return lattice, continuum


# ---------------- measurements ----------------
# A chain is either (n_sweeps, N) from metropolis, or (n_rows, n_walkers, N) from emcee.
def measure(chain, a, m, w):
    """Each observable for every stored path (averaged over time slices).
    Shape (n_sweeps,) for metropolis, (n_rows, n_walkers) for emcee."""
    dx = np.roll(chain, -1, axis=-1) - chain
    x2 = np.mean(chain**2, axis=-1)
    V = 0.5 * m * w**2 * x2
    T_prim = 1 / (2 * a) - m * np.mean(dx**2, axis=-1) / (2 * a**2)
    return {
        "x": np.mean(chain, axis=-1),
        "p": m * np.mean(dx, axis=-1) / a,              # telescopes to 0 on a periodic path
        "V": V,
        "T_prim": T_prim,
        "E_prim": T_prim + V,
        "E_vir": 2 * V,                                 # virial: <T> = 1/2 <x V'(x)> = <V>
        "x2": x2,
    }


def mean_and_error(series):
    """
    Metropolis (1-D series): binning error; flag says whether the binning curve plateaued.
    emcee (rows x walkers): each walker's time average is one estimate; error = spread of
    those / sqrt(n_walkers). Valid when the run is many autocorrelation times long and
    walkers are close to independent (true for large ensembles).
    """
    if series.ndim == 1:
        sizes, errs, errs_err = binning_error(series)
        plateau = abs(errs[-1] - errs[-2]) < np.hypot(errs_err[-1], errs_err[-2])
        return series.mean(), errs[-1], "yes" if plateau else "NO"
    walker_means = series.mean(axis=0)
    return series.mean(), walker_means.std(ddof=1) / np.sqrt(len(walker_means)), "walkers"


def autocorrelation(series, c=6.0):
    """
    Normalized autocorrelation rho(t) and integrated time tau_int = 1/2 + sum rho(t),
    summed up to the first window W with W >= c * tau_int(W) (Sokal's automatic window).
    For emcee (rows x walkers) the autocorrelation is averaged over walkers.
    Same convention as the binning estimate: err^2 = 2 tau_int var / n.
    """
    y = np.asarray(series, dtype=float)
    y = (y - y.mean(axis=0)).reshape(len(y), -1)        # (n, walkers); walkers = 1 for MH
    n = len(y)
    f = np.fft.rfft(y, 2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(axis=1)
    rho = acf / acf[0]
    taus = 0.5 + np.cumsum(rho[1:])
    W = next((t for t in range(1, n) if t >= c * taus[t - 1]), n - 1)
    return rho, taus[W - 1], W


# ---------------- analysis of one sampler ----------------
NAMES = {"x": "<x>", "p": "<p>", "V": "<V>", "T_prim": "<T> primitive",
         "E_prim": "<E> primitive", "E_vir": "<E> virial"}
LABELS = {"x": r"$\langle x\rangle$", "V": r"$\langle V\rangle$", "T_prim": r"$\langle T\rangle_{prim}$",
          "E_prim": r"$\langle E\rangle_{prim}$", "E_vir": r"$\langle E\rangle_{vir}$"}


def analyze(chain, acc_rate, burn, thin, a, m, w):
    """
    All measurements for one sampler.
    chain: metropolis (n_sweeps+1, N) or emcee (n_rows, n_walkers, N).
    burn: rows of the chain to discard (sweeps for metropolis, stored rows for emcee).
    thin: steps between stored rows (1 for metropolis); used to report times in steps.
    """
    kept = chain[burn:]
    obs = measure(kept, a, m, w)
    is_mh = kept.ndim == 2
    rho, tau, W = autocorrelation(obs["x2"])
    results = {k: mean_and_error(obs[k]) for k in NAMES}      # (mean, err, flag)
    vir = obs["T_prim"] - obs["V"]
    return {
        "kept": kept, "obs": obs, "results": results, "acc": acc_rate, "thin": thin,
        "is_mh": is_mh, "unit": "sweeps" if is_mh else "steps",
        "walkers": 1 if is_mh else kept.shape[1],
        "rho": rho, "tau": tau, "W": W,
        "n_eff": obs["x2"].size / (2 * tau),
        "run_in_tau": len(kept) / tau,
        "virial": (vir.mean(), mean_and_error(vir)[1]),
    }


# ---------------- combined report ----------------
def print_report(runs, N, a, m, w):
    lattice, continuum = exact_values(N, a, m, w)
    names = list(runs)
    print(f"N={N} a={a} m={m} w={w} beta={N * a:g}")
    for nm, r in runs.items():
        used = f"{len(r['kept']) * r['thin']} {r['unit']}"
        if not r["is_mh"]:
            used += f" x {r['walkers']} walkers"
        err_from = "binning" if r["is_mh"] else "spread between walkers"
        print(f"  {nm:11s} acc={r['acc']:.3f}  used={used}  "
              f"tau_int(x^2)={r['tau'] * r['thin']:.1f} {r['unit']}  "
              f"~{r['n_eff']:.0f} independent samples  errors: {err_from}")

    cell = 26
    head = f"\n{'observable':14s}" + "".join(f"{nm:>{cell}s}" for nm in names)
    head += f"{'lattice':>10s}{'continuum':>11s}"
    if len(names) == 2:
        head += f"{names[1][:5]}-{names[0][:4]}".rjust(14)
    print(head)
    no_plateau = False
    for k, label in NAMES.items():
        row = f"{label:14s}"
        for nm in names:
            mean, err, flag = runs[nm]["results"][k]
            if err < 1e-12:
                txt = f"{mean:+.4f} (trivial)"
            else:
                txt = f"{mean:.4f}±{err:.4f} {(mean - lattice[k]) / err:+.1f}σ"
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
    print("  σ columns: (MC - exact lattice) / err" +
          (f"; last column: ({names[1]} - {names[0]}) / combined err" if len(names) == 2 else ""))
    if no_plateau:
        print("  ! binning did not plateau: that error bar is too small; run longer")

    print(f"\nvirial check <T_prim> - <V>   (exact lattice "
          f"{lattice['T_prim'] - lattice['V']:+.4f}, continuum 0)")
    for nm, r in runs.items():
        print(f"  {nm:11s} {r['virial'][0]:+.4f} ± {r['virial'][1]:.4f}")

    for nm, r in runs.items():
        if r["n_eff"] < 1000:
            print(f"  warning ({nm}): fewer than ~1000 independent samples; run longer")
        if not r["is_mh"] and r["run_in_tau"] < 50:
            print(f"  warning ({nm}): run is only {r['run_in_tau']:.0f} tau long per walker "
                  "(want >= 50); walker-spread errors may be unreliable")


# ---------------- combined figures ----------------
def _pull_axes(ax, title, xlabel):
    ax.axvspan(-2, 2, color="0.5", alpha=0.12, label=r"$\pm 2\sigma$")
    ax.axvspan(-1, 1, color="0.5", alpha=0.25, label=r"$\pm 1\sigma$")
    ax.axvline(0, color="k", lw=1)
    ax.set(title=title, xlabel=xlabel)


def plot_comparison(runs, N, a, m, w, title=""):
    """
    Mirrors the table. With two samplers (hand-written first, emcee second):
      top left:     hand-written - exact lattice      (should sit on 0)
      top right:    emcee - exact lattice             (should sit on 0)
      bottom left:  emcee - hand-written, combined error (should sit on 0)
      bottom right: all three as pulls in sigma
    Continuum - lattice (red squares) shows the discretization error at finite a.
    With one sampler: its difference from lattice, and its pull.
    """
    lattice, continuum = exact_values(N, a, m, w)
    keys, names = list(LABELS), list(runs)
    y = np.arange(len(keys))
    gap = np.array([continuum[k] - lattice[k] for k in keys])

    def vals(nm):
        res = runs[nm]["results"]
        return (np.array([res[k][0] for k in keys]), np.array([res[k][1] for k in keys]))

    panels = []                                   # (title, diff, err, color, show continuum)
    for i, nm in enumerate(names):
        mean, err = vals(nm)
        lat = np.array([lattice[k] for k in keys])
        panels.append((f"{nm} − exact lattice", mean - lat, err, f"C{i}", True))
    if len(names) == 2:
        (ma, ea), (mb, eb) = vals(names[0]), vals(names[1])
        panels.append((f"{names[1]} − {names[0]}", mb - ma, np.hypot(ea, eb), "C4", False))

    ncols = 2
    nrows = int(np.ceil((len(panels) + 1) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), constrained_layout=True,
                           squeeze=False)
    ax = ax.ravel()
    for axi, (ttl, diff, err, col, show_cont) in zip(ax, panels):
        axi.axvline(0, color="k", lw=1)
        axi.errorbar(diff, y, xerr=err, fmt="o", capsize=4, color=col, label="difference ± err")
        if show_cont:
            axi.plot(gap, y, "s", mfc="none", color="C3", label="continuum − lattice")
        axi.set(yticks=y, yticklabels=[LABELS[k] for k in keys], title=ttl, xlabel="difference")
        axi.legend(fontsize=8)

    pax = ax[len(panels)]
    _pull_axes(pax, "Pulls (difference / error)", r"$\sigma$")
    off = np.linspace(0.2, -0.2, len(panels))
    lim = 3.5
    for (ttl, diff, err, col, _), o in zip(panels, off):
        pull = diff / err
        pax.plot(pull, y + o, "D" if col == "C4" else "o", color=col, label=ttl)
        lim = max(lim, np.abs(pull).max() + 0.5)
    pax.set(xlim=(-lim, lim), yticks=y, yticklabels=[LABELS[k] for k in keys])
    pax.legend(fontsize=8)
    for axi in ax[len(panels) + 1:]:
        axi.axis("off")
    fig.suptitle(title)
    return fig


def plot_distributions(runs, N, a, m, w, title=""):
    """
    Histograms after burn-in, all samplers overlaid. Panels 1-5: each observable for
    every stored path (width = path-to-path spread, NOT the error of the mean).
    Panel 6: every x_t pooled, vs exact lattice Gaussian and continuum |psi_0|^2.
    """
    lattice, _ = exact_values(N, a, m, w)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axi, k in zip(ax.flat, LABELS):
        for i, (nm, r) in enumerate(runs.items()):
            v = r["obs"][k].ravel()
            axi.hist(v, bins=60, density=True, histtype="step", lw=1.5, color=f"C{i}", label=nm)
            axi.axvline(v.mean(), color=f"C{i}", ls="--", lw=1)
        axi.axvline(lattice[k], color="k", lw=1.5, label="exact lattice")
        axi.set(title=LABELS[k] + " per path (dashed: MC mean)", xlabel="value", ylabel="density")
        axi.legend(fontsize=7)

    x2 = 2 * lattice["V"] / (m * w**2)                 # exact lattice <x^2>
    xs = np.linspace(-5 * np.sqrt(x2), 5 * np.sqrt(x2), 400)
    for i, (nm, r) in enumerate(runs.items()):
        ax[1, 2].hist(r["kept"].ravel(), bins=100, density=True, histtype="step", lw=1.5,
                      color=f"C{i}", label=nm)
    ax[1, 2].plot(xs, np.exp(-xs**2 / (2 * x2)) / np.sqrt(2 * np.pi * x2), "k", label="exact lattice")
    ax[1, 2].plot(xs, np.sqrt(m * w / np.pi) * np.exp(-m * w * xs**2), "C3--", label=r"continuum $|\psi_0|^2$")
    ax[1, 2].set(title="Position distribution", xlabel="x", ylabel="density")
    ax[1, 2].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def plot_autocorrelation(runs, title=""):
    """One row per sampler: rho(t) of x^2, and running tau_int vs window.
    Lags in sweeps (Metropolis) or steps (emcee): the units differ, so rows are separate."""
    fig, ax = plt.subplots(len(runs), 2, figsize=(11, 3.6 * len(runs)), constrained_layout=True,
                           squeeze=False)
    for i, (nm, r) in enumerate(runs.items()):
        rho, tau, W, th, unit = r["rho"], r["tau"], r["W"], r["thin"], r["unit"]
        lag = np.arange(min(len(rho), 5 * W))
        ax[i, 0].plot(lag * th, rho[lag], color=f"C{i}")
        ax[i, 0].axhline(0, color="k", lw=0.5)
        ax[i, 0].axvline(W * th, color="r", ls="--", label=f"window W={W * th}")
        ax[i, 0].set(title=rf"{nm}: autocorrelation of $x^2$", xlabel=f"lag ({unit})",
                     ylabel=r"$\rho(t)$")
        ax[i, 0].legend()
        taus = 0.5 + np.cumsum(rho[1:len(lag)])
        ax[i, 1].plot(lag[1:] * th, taus * th, color=f"C{i}")
        ax[i, 1].axhline(tau * th, color="r", ls="--", label=rf"$\tau_{{int}}$ = {tau * th:.1f}")
        ax[i, 1].set(title=f"{nm}: integrated autocorrelation time", xlabel=f"window ({unit})",
                     ylabel=rf"$\tau_{{int}}$ ({unit})")
        ax[i, 1].legend()
    fig.suptitle(title)
    return fig


def verify(runs, N, a, m, w, save=None):
    """
    runs: {name: (chain, acc_rate, burn, thin)} for one or more samplers, e.g.
          {"Metropolis": (chain, acc, burn, 1), "emcee": (chain_e, acc_e, burn_rows, thin)}
    Prints one combined table and draws combined comparison, distribution and
    autocorrelation figures. save: None, or a prefix like "run1" -> run1_comparison.png, ...
    Returns {name: {observable: (mean, err, flag)}}.
    """
    analyses = {nm: analyze(c, acc, burn, th, a, m, w) for nm, (c, acc, burn, th) in runs.items()}
    print_report(analyses, N, a, m, w)
    title = f"{' vs '.join(analyses)}: N={N}, a={a}, m={m}, w={w}"
    figs = {"comparison": plot_comparison(analyses, N, a, m, w, title),
            "distributions": plot_distributions(analyses, N, a, m, w, title),
            "autocorr": plot_autocorrelation(analyses, title)}
    if save:
        for name, f in figs.items():
            f.savefig(f"{save}_{name}.png", dpi=130)
    return {nm: r["results"] for nm, r in analyses.items()}


if __name__ == "__main__":
    # physics
    N = 100
    a = 0.5
    m = 1
    w = 1
    seed = None
    save = None          # or a prefix like "run1" -> run1_comparison.png, ...

    # hand-written Metropolis
    sweeps = 50000
    delta = 1
    x0 = 0
    burn = 1000          # sweeps

    # emcee: used ONLY as a reference to compare the hand-written sampler against. Decorrelates much more slowly
    # per step than the checkerboard Metropolis, so it needs many steps.
    # ~1-2 minutes and ~300 MB of memory with these settings.
    run_emcee = True
    n_walkers = 200      # must be >= 2 N
    steps = 100000
    thin = 50            # store every thin-th step
    x0_spread = 0.1      # walkers start at x0 + x0_spread * Normal(0, 1)
    burn_emcee = 20000   # steps

    chain, acc = metropolis(N, a, m, w, sweeps, delta, x0, seed)
    runs = {"Metropolis": (chain, acc, burn, 1)}
    if run_emcee:
        chain_e, acc_e = emcee_sampler(N, a, m, w, n_walkers, steps, thin, x0, x0_spread, seed)
        runs["emcee"] = (chain_e, acc_e, burn_emcee // thin, thin)

    verify(runs, N, a, m, w, save=save)
    if not save:
        plt.show()

r'''
MC = the simulation's estimate
\pm err = 1\sigma error bar from binning
lattice = the exact value 
dev = MC - lattice. anything within \pm 2\sigma is normal.
continuum = exact value for oscillator at temp \beta = Na
plateau = whether the binning curve levelled off, so thus it tells us if
          error bar is reliable
'''