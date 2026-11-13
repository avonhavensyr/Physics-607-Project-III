"""
Multi-walker validation of the lattice harmonic oscillator (sho_mcmc.py).

Runs K independent Metropolis walkers (own seed each, spread-out starting points)
and checks, for <x^2> and for the fraction of time spent in each region [lo, hi]:

  a) each walker vs the exact lattice value      pulls, chi^2 / K  (should be ~1)
  b) walkers vs each other                       Gelman-Rubin R-hat (should be < ~1.01)
  c) error bars are honest                       scatter of walker means / typical
                                                 binning error     (should be ~1)
  d) burn-in is long enough                      trace plot of x^2 for every walker

Region fraction: every x_t of the path is a Gaussian with variance C(0) = exact
lattice <x^2>, so the expected fraction in [lo, hi] comes from erf.
MC estimate per sweep: the fraction of the N time slices inside [lo, hi].

Set the parameters at the bottom, then run:  python verify_sho_ALEX.py
"""
from math import erf, sqrt

import numpy as np
import matplotlib.pyplot as plt
from sho_mcmc import metropolis, binning_error
from verify_sho import exact_correlator


# ---------------- exact results ----------------
def gaussian_fraction(lo, hi, var):
    """P(lo <= x <= hi) for x ~ N(0, var). lo/hi may be -inf/inf."""
    cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2 * var))) if np.isfinite(x) else float(x > 0)
    return cdf(hi) - cdf(lo)


def exact_region(regions, N, a, m, w):
    """Exact lattice and continuum (beta = N a) <x^2> and region fractions."""
    x2_lat = exact_correlator(N, a, m, w)[0]
    x2_con = 1 / (2 * m * w * np.tanh(N * a * w / 2))
    lattice = {"x2": x2_lat}
    continuum = {"x2": x2_con}
    for lo, hi in regions:
        lattice[(lo, hi)] = gaussian_fraction(lo, hi, x2_lat)
        continuum[(lo, hi)] = gaussian_fraction(lo, hi, x2_con)
    return lattice, continuum


# ---------------- walkers ----------------
def run_walkers(K, N, a, m, w, sweeps, delta, x0_spread=3.0, seed=None):
    """K independent chains with starting points spread over [-x0_spread, x0_spread]."""
    x0s = np.linspace(-x0_spread, x0_spread, K)
    seeds = np.random.SeedSequence(seed).spawn(K)
    chains, accs = [], []
    for x0, s in zip(x0s, seeds):
        chain, acc = metropolis(N, a, m, w, sweeps, delta, x0, s)
        chains.append(chain)
        accs.append(acc)
    return np.array(chains), np.array(accs), x0s      # chains: (K, sweeps+1, N)


def per_sweep(chain, regions):
    """One value per sweep: <x^2> over time slices, and fraction of slices in each region."""
    obs = {"x2": np.mean(chain**2, axis=1)}
    for lo, hi in regions:
        obs[(lo, hi)] = np.mean((chain >= lo) & (chain <= hi), axis=1)
    return obs


def gelman_rubin(series):
    """R-hat for series of shape (K, n): between- vs within-walker variance."""
    K, n = series.shape
    W = series.var(axis=1, ddof=1).mean()
    B = n * series.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def name(key):
    if key == "x2":
        return "<x^2>"
    lo, hi = key
    return f"frac [{lo:g}, {hi:g}]"


# ---------------- report ----------------
def verify_walkers(chains, accs, x0s, regions, N, a, m, w, burn, save=None):
    """Print the multi-walker table and draw trace / pull / region figures."""
    K = len(chains)
    obs = [per_sweep(c[burn:], regions) for c in chains]
    lattice, continuum = exact_region(regions, N, a, m, w)
    keys = list(lattice)

    print(f"N={N} a={a} m={m} w={w} beta={N * a:g}  K={K} walkers  "
          f"sweeps used={chains.shape[1] - 1 - burn} each  "
          f"acc={accs.mean():.3f} (range {accs.min():.3f}-{accs.max():.3f})")
    print(f"{'observable':18s} {'MC (pooled)':>11s} {'± err':>8s} {'lattice':>9s} {'dev':>7s} "
          f"{'continuum':>10s} {'chi2/K':>7s} {'in 2σ':>6s} {'R-hat':>7s} {'scatter/err':>11s}")

    results = {}
    for k in keys:
        series = np.array([o[k] for o in obs])                     # (K, n)
        means = series.mean(axis=1)
        errs = np.array([binning_error(s)[1][-1] for s in series])
        pulls = (means - lattice[k]) / errs
        chi2_K = np.mean(pulls**2)
        in2 = np.mean(np.abs(pulls) < 2)
        rhat = gelman_rubin(series)
        scatter = means.std(ddof=1) / np.sqrt(np.mean(errs**2))
        # pooled estimate: walkers are independent, so errors add in quadrature
        pooled, pooled_err = means.mean(), np.sqrt(np.sum(errs**2)) / K
        dev = (pooled - lattice[k]) / pooled_err
        print(f"{name(k):18s} {pooled:11.4f} {pooled_err:8.4f} {lattice[k]:9.4f} {dev:+6.1f}σ "
              f"{continuum[k]:10.4f} {chi2_K:7.2f} {in2:6.0%} {rhat:7.4f} {scatter:11.2f}")
        results[k] = {"means": means, "errs": errs, "pulls": pulls, "chi2_K": chi2_K,
                      "rhat": rhat, "scatter": scatter, "pooled": (pooled, pooled_err)}

    bad = [name(k) for k in keys if results[k]["rhat"] > 1.01]
    if bad:
        print(f"  warning: R-hat > 1.01 for {', '.join(bad)}; walkers not converged, "
              "increase sweeps or burn")
    print("expect: dev within ±2σ, chi2/K ~ 1, ~95% of walkers in 2σ, R-hat < 1.01, scatter/err ~ 1")

    title = f"N={N}, a={a}, m={m}, w={w}, K={K} walkers"
    figs = {"traces": plot_traces(chains, x0s, lattice, burn, title),
            "pulls": plot_pulls(results, keys, title),
            "regions": plot_regions(results, regions, lattice, continuum, title)}
    if save:
        for fname, f in figs.items():
            f.savefig(f"{save}_{fname}.png", dpi=130)
    return results


# ---------------- plots ----------------
def plot_traces(chains, x0s, lattice, burn, title=""):
    """x^2 per sweep for every walker, including burn-in (check d)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x2 = np.mean(chains**2, axis=2)                                # (K, sweeps+1)
    early = min(x2.shape[1], 3 * burn) if burn else x2.shape[1]
    for i, (tr, x0) in enumerate(zip(x2, x0s)):
        ax[0].plot(tr[:early], lw=0.7, label=f"x0={x0:+.1f}")
        ax[1].plot(tr, lw=0.4, alpha=0.6, color=f"C{i % 10}")
    for axi in ax:
        axi.axhline(lattice["x2"], color="k", lw=1.2, label="exact lattice")
        axi.axvline(burn, color="r", ls="--", label="burn")
        axi.set(xlabel="sweep", ylabel=r"$x^2$ (avg over slices)")
    ax[0].set(title="Start of run (burn-in)", yscale="log")
    ax[1].set(title="Whole run")
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    return fig


def plot_pulls(results, keys, title=""):
    """Pull of every walker for every observable, with 1 and 2 sigma bands (check a)."""
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(keys) + 1.5), constrained_layout=True)
    ax.axvspan(-2, 2, color="C2", alpha=0.12, label=r"$\pm 2\sigma$")
    ax.axvspan(-1, 1, color="C2", alpha=0.25, label=r"$\pm 1\sigma$")
    ax.axvline(0, color="k", lw=1)
    for j, k in enumerate(keys):
        p = results[k]["pulls"]
        ax.plot(p, np.full_like(p, j) + np.linspace(-0.15, 0.15, len(p)), "o", ms=4)
    lim = max(3.5, max(np.abs(results[k]["pulls"]).max() for k in keys) + 0.5)
    ax.set(yticks=range(len(keys)), yticklabels=[name(k) for k in keys], xlim=(-lim, lim),
           xlabel=r"(walker − lattice) / err  [$\sigma$]", title="Per-walker pulls")
    ax.legend(fontsize=8)
    fig.suptitle(title)
    return fig


def plot_regions(results, regions, lattice, continuum, title=""):
    """Fraction of time in each region: every walker, pooled, exact lattice, continuum."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for j, r in enumerate(regions):
        res = results[r]
        K = len(res["means"])
        ax.errorbar(j + np.linspace(-0.25, 0.05, K), res["means"], yerr=res["errs"],
                    fmt="o", ms=3, color="C0", alpha=0.7, label="walkers" if j == 0 else None)
        ax.errorbar(j + 0.15, *res["pooled"], fmt="D", color="C1", capsize=4,
                    label="pooled" if j == 0 else None)
        ax.hlines(lattice[r], j - 0.35, j + 0.35, color="k", label="exact lattice" if j == 0 else None)
        ax.hlines(continuum[r], j - 0.35, j + 0.35, color="C3", ls="--",
                  label="continuum" if j == 0 else None)
    ax.set(xticks=range(len(regions)), xticklabels=[name(r) for r in regions],
           ylabel="fraction of time in region", title="Time spent in each region")
    ax.legend(fontsize=8)
    fig.suptitle(title)
    return fig


if __name__ == "__main__":
    N = 100
    a = 0.5
    m = 1
    w = 1
    K = 6                # number of independent walkers
    sweeps = 20000       # per walker
    delta = 1
    x0_spread = 3.0      # walkers start at x0 evenly spaced in [-x0_spread, x0_spread]
    burn = 1000
    seed = None
    save = None          # or a prefix like "alex1" -> alex1_traces.png, alex1_pulls.png, ...

    # regions [lo, hi] in x; use np.inf for an open-ended tail
    regions = [(-0.5, 0.5), (0.5, 1.0), (1.0, np.inf), (-np.inf, -1.0)]

    chains, accs, x0s = run_walkers(K, N, a, m, w, sweeps, delta, x0_spread, seed)
    verify_walkers(chains, accs, x0s, regions, N, a, m, w, burn, save=save)
    if not save:
        plt.show()
