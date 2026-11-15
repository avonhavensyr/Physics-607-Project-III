"""
Off-the-shelf sampler: emcee on the same lattice target (Kevin's code).

emcee's affine-invariant ensemble sampler (stretch move) samples
P[x] ~ exp(-S[x]) with log P = -S, evaluated for all walkers at once. It is used
only as a reference to compare the hand-written Metropolis against (Test 9).
"""

import emcee
import numpy as np

from .lattice import action


def emcee_sampler(N, a, m, w, n_walkers, n_steps, thin, x0, x0_spread, seed=None):
    """
    Sample exp(-S[x]) with emcee's ensemble sampler.

    Parameters
    ----------
    N : int
        Number of time slices, i.e. the dimension of the target.
    a, m, w : float
        Lattice spacing, mass and frequency.
    n_walkers : int
        Number of walkers; must be at least 2 N.
    n_steps : int
        Number of steps per walker.
    thin : int
        Every `thin`-th step is stored.
    x0 : float
        Centre of the starting points.
    x0_spread : float
        Walkers start at x0 + x0_spread * Normal(0, 1); they must not all start
        at the same point.
    seed : int or None, optional
        Random seed.

    Returns
    -------
    chain : ndarray, shape (n_steps // thin, n_walkers, N)
        Stored positions of every walker. Memory: (n_steps // thin) * n_walkers
        * N * 8 bytes.
    acc_rate : float
        Mean acceptance fraction over the walkers.
    """
    sampler = emcee.EnsembleSampler(
        n_walkers, N, lambda X: -action(X, a, m, w), vectorize=True
    )
    if seed is not None:
        sampler.random_state = np.random.RandomState(seed).get_state()
    p0 = x0 + x0_spread * np.random.default_rng(seed).standard_normal((n_walkers, N))
    # stores n_steps // thin rows
    sampler.run_mcmc(
        p0,
        n_steps // thin,
        thin_by=thin,
        progress=True,
        progress_kwargs={"desc": "Emcee", "unit": "step"},
    )
    return sampler.get_chain(), sampler.acceptance_fraction.mean()
