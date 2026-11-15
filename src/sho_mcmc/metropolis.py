"""Hand-written checkerboard Metropolis sampler for the lattice oscillator."""
import numpy as np

from .lattice import coefficients


def metropolis(N, a, m, w, n_sweeps, delta, x0, seed=None):
    """
    Checkerboard Metropolis sampling of exp(-S[x]) on a periodic lattice.

    Sites of the same parity are not neighbours, so all even sites are updated
    at once, then all odd sites. One sweep is both halves, i.e. one proposal
    per site.

    Parameters
    ----------
    N : int
        Number of time slices; must be even for the checkerboard update.
    a, m, w : float
        Lattice spacing, mass and frequency.
    n_sweeps : int
        Number of sweeps.
    delta : float
        Proposal half-width: each proposal is x_t + u with u uniform in
        [-delta, delta].
    x0 : float or array_like, shape (N,)
        Starting path. A single number starts every x_t at that value.
    seed : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    chain : ndarray, shape (n_sweeps + 1, N)
        The starting path followed by the path after every sweep.
    acc_rate : float
        Fraction of accepted proposals.

    Raises
    ------
    ValueError
        If N is odd.
    """
    if N % 2:
        raise ValueError("N must be even for the checkerboard update")
    A, B = coefficients(a, m, w)
    rng = np.random.default_rng(seed)
    x = np.broadcast_to(np.asarray(x0, dtype=float), (N,)).copy()
    chain = np.empty((n_sweeps + 1, N))
    chain[0] = x
    even = np.arange(N) % 2 == 0
    accepted = 0

    for s in range(1, n_sweeps + 1):
        for half in (even, ~even):
            shift = rng.uniform(-delta, delta, N)
            # change of the action when x_t -> x_t + shift; it depends only on
            # the two neighbours, which belong to the other half
            dS = shift * (A * (shift + 2 * x) - B * (np.roll(x, 1) + np.roll(x, -1)))
            # accept with probability min(1, exp(-dS)), only on this half
            acc = half & (np.log(rng.random(N)) < -dS)
            x = x + shift * acc
            accepted += acc.sum()
        chain[s] = x

    return chain, accepted / (n_sweeps * N)
