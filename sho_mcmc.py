"""
Euclidean path-integral Metropolis for the harmonic oscillator (hbar = 1).

The path is one array x of length N: x[t] = position at time slice t (spacing a),
with periodic ends. Lattice action:
    S[x] = sum_t [ A x_t^2 - B x_t x_{t+1} ],   A = m/a + a m w^2/2,  B = m/a
We sample P[x] ~ exp(-S[x]).

Usage:
    from sho_mcmc import metropolis, exact_x2, summary, plot_diagnostics
    chain, acc = metropolis(N=..., a=..., m=..., w=..., n_sweeps=..., delta=..., x0=..., seed=...)
    summary(chain, acc, N=..., a=..., m=..., w=..., burn=..., filename=...)
    plot_diagnostics(chain, a=..., m=..., w=..., burn=..., thin=...)

    # off-the-shelf sampler (emcee) on the same target
    from sho_mcmc import emcee_sampler
    chain, acc = emcee_sampler(N=..., a=..., m=..., w=..., n_walkers=..., n_steps=..., thin=...,
                               x0=..., x0_spread=..., seed=...)
    # chain shape: (n_steps // thin, n_walkers, N). summary/plot_diagnostics expect a
    # metropolis chain (n_sweeps+1, N); verify_sho.verify handles both.
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee
import tqdm



def coefficients(a, m, w):
    """A, B of the lattice action."""
    return m / a + a * m * w**2 / 2, m / a


def action(chain, a, m, w):
    """S[x] for one path (shape N) or every path in a chain (shape n, N)."""
    A, B = coefficients(a, m, w)
    return A * np.sum(chain**2, -1) - B * np.sum(chain * np.roll(chain, -1, -1), -1)


def metropolis(N, a, m, w, n_sweeps, delta, x0, seed=None):
    """
    Checkerboard Metropolis on a periodic time lattice of N sites (N even).
    Sites of the same parity don't share neighbors, so all even sites are updated
    at once, then all odd sites. One sweep = both halves.
    Returns chain (n_sweeps+1, N) and the acceptance rate.
    """
    if N % 2:
        raise ValueError("N must be even for the checkerboard update")
    A, B = coefficients(a, m, w)
    rng = np.random.default_rng(seed)
    x = np.full(N, float(x0))
    chain = np.empty((n_sweeps + 1, N))
    chain[0] = x
    even = np.arange(N) % 2 == 0
    accepted = 0

    for s in tqdm.tqdm(range(1, n_sweeps + 1)):
        for half in (even, ~even):
            shift = rng.uniform(-delta, delta, N)
            dS = shift * (A * (shift + 2 * x) - B * (np.roll(x, 1) + np.roll(x, -1)))
            acc = half & (np.log(rng.random(N)) < -dS)        # prob min(1, e^{-dS})
            x = x + shift * acc
            accepted += acc.sum()
        chain[s] = x

    return chain, accepted / (n_sweeps * N)


def emcee_sampler(N, a, m, w, n_walkers, n_steps, thin, x0, x0_spread, seed=None):
    """
    Same target P[x] ~ exp(-S[x]) sampled with emcee's affine-invariant ensemble
    (stretch move). log P = -S, evaluated for all walkers at once.
    n_walkers must be >= 2 N. Walkers start at x0 + x0_spread * Normal(0, 1)
    (they must not all start at the same point). Every `thin`-th step is stored.
    Memory: (n_steps // thin) * n_walkers * N * 8 bytes.
    Returns chain (n_steps // thin, n_walkers, N) and the mean acceptance fraction.
    """
    sampler = emcee.EnsembleSampler(n_walkers, N, lambda X: -action(X, a, m, w), vectorize=True)
    if seed is not None:
        sampler.random_state = np.random.RandomState(seed).get_state()
    p0 = x0 + x0_spread * np.random.default_rng(seed).standard_normal((n_walkers, N))
    sampler.run_mcmc(p0, n_steps // thin, thin_by=thin)   # stores n_steps // thin rows
    return sampler.get_chain(), sampler.acceptance_fraction.mean()


def exact_x2(N, a, m, w):
    """Exact lattice <x^2> on the periodic lattice, from the eigenvalues of S."""
    A, B = coefficients(a, m, w)
    k = np.arange(N)
    return np.mean(1 / (2 * A - 2 * B * np.cos(2 * np.pi * k / N)))


def binning_error(series, min_blocks=32):
    """
    Error of the mean of a correlated time series by binning.
    Block sizes 1, 2, 4, ... while at least `min_blocks` blocks remain.
    Returns (block_sizes, errors, errors_err); the error at the largest block
    size is the estimate, valid once the curve has reached a plateau.
    errors_err is the statistical uncertainty of each error estimate.
    """
    series = np.asarray(series)
    sizes, errs, errs_err = [], [], []
    b = 1
    while len(series) // b >= min_blocks:
        nb = len(series) // b
        blocks = series[: nb * b].reshape(nb, b).mean(1)
        e = blocks.std(ddof=1) / np.sqrt(nb)
        sizes.append(b); errs.append(e); errs_err.append(e / np.sqrt(2 * (nb - 1)))
        b *= 2
    return np.array(sizes), np.array(errs), np.array(errs_err)


def summary(chain, acc_rate, N, a, m, w, burn, filename=None):
    """
    MC <x^2> with a binning error bar, next to the exact lattice and continuum values.
    Uses every sweep after `burn` (no thinning needed: binning handles correlations).
    Draws error vs block size; the error is trustworthy once that curve plateaus.
    Returns (mc, err, lattice, continuum).
    """
    series = np.mean(chain[burn:] ** 2, axis=1)       # one <x^2> per sweep
    mc = series.mean()
    sizes, errs, errs_err = binning_error(series)
    err = errs[-1]
    tau = 0.5 * (err / errs[0]) ** 2                  # integrated autocorrelation time (sweeps)
    lat, cont = exact_x2(N, a, m, w), 1 / (2 * m * w)
    print(f"N={N} a={a} m={m} w={w}  acc={acc_rate:.3f}  tau_int~{tau:.1f} sweeps\n"
          f"  <x^2>: MC={mc:.4f} ± {err:.4f}  lattice={lat:.4f} ({(mc - lat) / err:+.1f} sigma)"
          f"  continuum={cont:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.errorbar(sizes, errs, yerr=errs_err, marker="o", capsize=3)
    ax.set(xscale="log", xlabel="block size (sweeps)", ylabel=r"error of $\langle x^2\rangle$",
           title=f"Binning analysis (N={N}, a={a}, w={w})")
    if filename:
        fig.savefig(filename, dpi=130)
    return mc, err, lat, cont


def plot_diagnostics(chain, a, m, w, burn, thin, t0=None, filename=None):
    """Walker trace, histogram vs |psi_0|^2, four path samples, action history."""
    n_sweeps, N = chain.shape[0] - 1, chain.shape[1]
    t0 = N // 2 if t0 is None else t0
    samples = chain[burn::thin].ravel()       # every site is a draw from |psi_0|^2
    x2_lat = exact_x2(N, a, m, w)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax[0, 0].plot(chain[:, t0], lw=0.5)
    ax[0, 0].axvline(burn, color="k", ls="--", label="burn-in")
    ax[0, 0].set(title="Walker evolution", xlabel="Sweep", ylabel=f"$x_{{{t0}}}$")
    ax[0, 0].legend()

    ax[0, 1].hist(samples, bins=100, density=True, alpha=0.6, label="MCMC")
    r = 6 * np.sqrt(x2_lat)
    xs = np.linspace(-r, r, 400)
    ax[0, 1].plot(xs, np.exp(-xs**2 / (2 * x2_lat)) / np.sqrt(2 * np.pi * x2_lat), label="exact lattice")
    ax[0, 1].plot(xs, np.sqrt(m * w / np.pi) * np.exp(-m * w * xs**2), "--", label=r"continuum $|\psi_0|^2$")
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
