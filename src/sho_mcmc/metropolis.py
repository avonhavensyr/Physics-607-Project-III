"""Hand-written checkerboard Metropolis sampler for the lattice oscillator."""

import numpy as np
from scipy.special import erfc
from tqdm import tqdm

from .lattice import coefficients


def delta_action(x, shift, A, B):
    """
    Change of the action when each x_t alone is moved to x_t + shift_t.

    Only the two neighbours of a site enter, so the value for site t is exact
    while its neighbours stay fixed, as in the checkerboard update.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Current path.
    shift : ndarray, shape (N,)
        Proposed change at every site.
    A, B : float
        Coefficients of the action (see `sho_mcmc.lattice.coefficients`).

    Returns
    -------
    ndarray, shape (N,)
        dS_t = shift_t [A (shift_t + 2 x_t) - B (x_{t-1} + x_{t+1})].
    """
    return shift * (A * (shift + 2 * x) - B * (np.roll(x, 1) + np.roll(x, -1)))


def expected_acceptance(delta, a, m, w):
    """
    Exact average acceptance rate of the update (Eq. (8) of the plan).

    Given its neighbours, x_t is Gaussian with variance 1/(2A), so the average
    acceptance depends only on D = delta sqrt(A), the step in the plan's units.

    Parameters
    ----------
    delta : float or ndarray
        Proposal half-width in the code's units.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    float or ndarray
        erfc(D/2) + 2 / (sqrt(pi) D) (1 - exp(-D^2/4)).
    """
    A, _ = coefficients(a, m, w)
    D = np.asarray(delta) * np.sqrt(A)
    return erfc(D / 2) + 2 / (np.sqrt(np.pi) * D) * (1 - np.exp(-(D**2) / 4))


def metropolis(
    N, a, m, w, n_sweeps, delta, x0, seed=None, history=False, progress=True
):
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
    history : bool, optional
        If True, also return the acceptance rate of every sweep.
    progress : bool, optional
        Show a progress bar (Kevin's addition). The tests switch it off.

    Returns
    -------
    chain : ndarray, shape (n_sweeps + 1, N)
        The starting path followed by the path after every sweep.
    acc_rate : float
        Fraction of accepted proposals.
    acc_history : ndarray, shape (n_sweeps,)
        Acceptance rate of each sweep; only returned if `history` is True.

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
    acc_history = np.empty(n_sweeps)

    sweeps = tqdm(
        range(1, n_sweeps + 1), desc="Metropolis", unit="sweep", disable=not progress
    )
    for s in sweeps:
        n_acc = 0
        for half in (even, ~even):
            shift = rng.uniform(-delta, delta, N)
            # the change of the action depends only on the two neighbours,
            # which belong to the other half and stay fixed here
            dS = delta_action(x, shift, A, B)
            # accept with probability min(1, exp(-dS)), only on this half
            acc = half & (np.log(rng.random(N)) < -dS)
            x = x + shift * acc
            n_acc += acc.sum()
        chain[s] = x
        acc_history[s - 1] = n_acc / N

    acc_rate = acc_history.mean()
    if history:
        return chain, acc_rate, acc_history
    return chain, acc_rate
