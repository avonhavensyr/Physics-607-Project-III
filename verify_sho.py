

"""
Physics verifications for the lattice harmonic oscillator, using sho_simple.py.

Set the parameters in the block at the bottom, then run:  python verify_sho.py

Checks (each MC value has a binning error bar):
  <x>  = 0                                   by symmetry
  <p>  = 0                                   by symmetry (identically 0 on a periodic lattice)
  <V>  = 1/2 m w^2 <x^2>                     vs exact lattice and continuum at beta = N a
  <E>  primitive estimator                   vs exact lattice and continuum
  <E>  virial estimator  (<T> = <V> for SHO) vs continuum
  virial theorem: <T_primitive> vs <V>
  autocorrelation function of x^2 and its integrated autocorrelation time
"""
import numpy as np
import matplotlib.pyplot as plt
from sho_mcmc import metropolis, coefficients, binning_error
import tqdm


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
def measure(chain, a, m, w):
    """One value per sweep of each observable (averaged over time slices)."""
    dx = np.roll(chain, -1, axis=1) - chain
    x2 = np.mean(chain**2, axis=1)
    V = 0.5 * m * w**2 * x2
    T_prim = 1 / (2 * a) - m * np.mean(dx**2, axis=1) / (2 * a**2)
    return {
        "x": np.mean(chain, axis=1),
        "p": m * np.mean(dx, axis=1) / a,               # telescopes to 0 on a periodic path
        "V": V,
        "T_prim": T_prim,
        "E_prim": T_prim + V,
        "E_vir": 2 * V,                                 # virial: <T> = 1/2 <x V'(x)> = <V>
        "x2": x2,
    }


def autocorrelation(series, c=6.0):
    """
    Normalized autocorrelation rho(t) and integrated time tau_int = 1/2 + sum rho(t),
    summed up to the first window W with W >= c * tau_int(W) (Sokal's automatic window).
    Same convention as the binning estimate: err^2 = 2 tau_int var / n.
    """
    y = np.asarray(series) - np.mean(series)
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    rho = acf / acf[0]
    taus = 0.5 + np.cumsum(rho[1:])
    W = next((t for t in range(1, n) if t >= c * taus[t - 1]), n - 1)
    return rho, taus[W - 1], W


# ---------------- comparison plots ----------------
LABELS = {"x": r"$\langle x\rangle$", "V": r"$\langle V\rangle$", "T_prim": r"$\langle T\rangle_{prim}$",
          "E_prim": r"$\langle E\rangle_{prim}$", "E_vir": r"$\langle E\rangle_{vir}$"}


def plot_comparison(results, lattice, continuum, title=""):
    """
    Left: MC - lattice with error bars (should sit on 0), and continuum - lattice
    (the discretization error, expected to be nonzero at finite a).
    Right: pull (MC - lattice)/err with 1 and 2 sigma bands.
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
    ax[0].set(yticks=y, yticklabels=[LABELS[k] for k in keys], xlabel="value − exact lattice",
              title="MC and continuum relative to exact lattice")
    ax[0].legend(fontsize=8)

    pull = diff / err
    ax[1].axvspan(-2, 2, color="C2", alpha=0.12, label=r"$\pm 2\sigma$")
    ax[1].axvspan(-1, 1, color="C2", alpha=0.25, label=r"$\pm 1\sigma$")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].plot(pull, y, "o")
    lim = max(3.5, np.abs(pull).max() + 0.5)
    ax[1].set(xlim=(-lim, lim), xlabel=r"(MC − lattice) / err  [$\sigma$]", title="Pull")
    ax[1].legend(fontsize=8)
    fig.suptitle(title)
    return fig


def plot_distributions(chain, obs, lattice, m, w, title=""):
    """
    Histograms after burn-in. Panels 1-5: per-sweep values of each observable
    (their width is the sweep-to-sweep spread, NOT the error of the mean).
    Panel 6: every x_t pooled, vs exact lattice Gaussian and continuum |psi_0|^2.
    """
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axi, k in zip(ax.flat, LABELS):
        axi.hist(obs[k], bins=60, density=True, alpha=0.6)
        axi.axvline(lattice[k], color="k", lw=1.5, label="exact lattice")
        axi.axvline(obs[k].mean(), color="C1", ls="--", lw=1.5, label="MC mean")
        axi.set(title=LABELS[k] + " per sweep", xlabel="value", ylabel="density")
        axi.legend(fontsize=7)

    x2 = 2 * lattice["V"] / (m * w**2)                 # exact lattice <x^2>
    xs = np.linspace(-5 * np.sqrt(x2), 5 * np.sqrt(x2), 400)
    ax[1, 2].hist(chain.ravel(), bins=100, density=True, alpha=0.6, label="MC, all $x_t$")
    ax[1, 2].plot(xs, np.exp(-xs**2 / (2 * x2)) / np.sqrt(2 * np.pi * x2), "k", label="exact lattice")
    ax[1, 2].plot(xs, np.sqrt(m * w / np.pi) * np.exp(-m * w * xs**2), "C3--", label=r"continuum $|\psi_0|^2$")
    ax[1, 2].set(title="Position distribution", xlabel="x", ylabel="density")
    ax[1, 2].legend(fontsize=7)
    fig.suptitle(title)
    return fig


# ---------------- report ----------------
def verify(chain, acc_rate, N, a, m, w, burn, save=None):
    """Print the table and draw comparison, distribution and autocorrelation figures.
    save: None to only draw, or a prefix like "run1" -> run1_comparison.png, etc."""
    obs = measure(chain[burn:], a, m, w)
    lattice, continuum = exact_values(N, a, m, w)
    labels = {"x": "<x>", "p": "<p>", "V": "<V>", "T_prim": "<T> primitive",
              "E_prim": "<E> primitive", "E_vir": "<E> virial"}

    print(f"N={N} a={a} m={m} w={w} beta={N * a:g}  acc={acc_rate:.3f}  "
          f"sweeps used={len(chain) - 1 - burn}")
    print(f"{'observable':15s} {'MC':>9s} {'± err':>8s} {'lattice':>9s} {'dev':>7s} "
          f"{'continuum':>10s}  plateau?")
    results = {}
    for k, name in labels.items():
        mean = obs[k].mean()
        sizes, errs, errs_err = binning_error(obs[k])
        err = errs[-1]
        # plateau check: last two block sizes agree within their errors
        plateau = abs(errs[-1] - errs[-2]) < np.hypot(errs_err[-1], errs_err[-2])
        if err < 1e-12:                                 # e.g. <p>: zero for every path
            dev, flag = "exact 0", "trivial"
        else:
            dev, flag = f"{(mean - lattice[k]) / err:+.1f}σ", "yes" if plateau else "NO"
        print(f"{name:15s} {mean:9.4f} {err:8.4f} {lattice[k]:9.4f} {dev:>7s} "
              f"{continuum[k]:10.4f}  {flag}")
        results[k] = (mean, err)

    # virial theorem: <T> should equal <V> (exactly in the continuum, O(a^2) on the lattice)
    diff = obs["T_prim"] - obs["V"]
    d_err = binning_error(diff)[1][-1]
    print(f"\nvirial check <T_prim> - <V> = {diff.mean():+.4f} ± {d_err:.4f}"
          f"   (exact lattice {lattice['T_prim'] - lattice['V']:+.4f}, continuum 0)")

    # autocorrelation of x^2
    rho, tau, W = autocorrelation(obs["x2"])
    tau_bin = 0.5 * (binning_error(obs["x2"])[1][-1] / obs["x2"].std(ddof=1)
                     * np.sqrt(len(obs["x2"]))) ** 2
    n_eff = len(obs["x2"]) / (2 * tau)
    print(f"x^2 autocorrelation: tau_int = {tau:.1f} sweeps (window {W}), "
          f"binning gives {tau_bin:.1f};  ~{n_eff:.0f} independent samples")
    if n_eff < 1000:
        print("  warning: fewer than ~1000 independent samples; increase --sweeps")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    lag = np.arange(min(len(rho), 5 * W))
    ax[0].plot(lag, rho[lag])
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axvline(W, color="r", ls="--", label=f"window W={W}")
    ax[0].set(title=r"Autocorrelation of $x^2$", xlabel="lag (sweeps)", ylabel=r"$\rho(t)$")
    ax[0].legend()
    taus = 0.5 + np.cumsum(rho[1:len(lag)])
    ax[1].plot(lag[1:], taus)
    ax[1].axhline(tau, color="r", ls="--", label=rf"$\tau_{{int}}$ = {tau:.1f}")
    ax[1].set(title=r"Integrated autocorrelation time vs window", xlabel="window (sweeps)",
              ylabel=r"$\tau_{int}$")
    ax[1].legend()

    title = f"N={N}, a={a}, m={m}, w={w}"
    figs = {"autocorr": fig,
            "comparison": plot_comparison(results, lattice, continuum, title),
            "distributions": plot_distributions(chain[burn:], obs, lattice, m, w, title)}
    if save:
        for name, f in figs.items():
            f.savefig(f"{save}_{name}.png", dpi=130)
    return results


if __name__ == "__main__":
    N = 100
    a = 0.5
    m = 1
    w = 1
    sweeps = 50000
    delta = 1
    x0 = 0
    burn = 1000
    seed = None
    save = None          # or a prefix like "run1" -> run1_comparison.png, run1_distributions.png, ...

    chain, acc = metropolis(N, a, m, w, sweeps, delta, x0, seed)
    verify(chain, acc, N, a, m, w, burn, save=save)
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