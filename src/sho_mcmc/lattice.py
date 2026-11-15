"""
Lattice action of the harmonic oscillator and its exact results.

The path x_0, ..., x_{N-1} lives on a periodic time lattice with spacing a, and
the Euclidean action is (hbar = 1)

    S[x] = sum_t [ m/(2a) (x_{t+1} - x_t)^2 + a m w^2 x_t^2 / 2 ]
         = A sum_t x_t^2 - B sum_t x_t x_{t+1},

with A = m/a + a m w^2 / 2 and B = m/a. Paths are sampled from exp(-S[x]), a
Gaussian, so many averages are known exactly.
"""
import numpy as np


def coefficients(a, m, w):
    """
    Coefficients of the lattice action.

    Parameters
    ----------
    a : float
        Lattice spacing.
    m : float
        Mass.
    w : float
        Oscillator frequency.

    Returns
    -------
    A : float
        Coefficient of sum_t x_t^2, ``m/a + a m w^2 / 2``.
    B : float
        Coefficient of -sum_t x_t x_{t+1}, ``m/a``.
    """
    return m / a + a * m * w**2 / 2, m / a


def action(chain, a, m, w):
    """
    Lattice action S[x] of one path or of many paths at once.

    Parameters
    ----------
    chain : ndarray, shape (N,) or (n, N)
        One path, or n paths with N time slices each.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    float or ndarray, shape (n,)
        The action of each path.
    """
    A, B = coefficients(a, m, w)
    # np.roll(..., -1, -1) gives x_{t+1}, with x_N = x_0 (periodic lattice)
    return A * np.sum(chain**2, -1) - B * np.sum(chain * np.roll(chain, -1, -1), -1)


def exact_correlator(N, a, m, w):
    """
    Exact correlator C(j) = <x_t x_{t+j}> on the periodic lattice.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    ndarray, shape (N,)
        C(j) for j = 0, ..., N-1, from the eigenvalues 2A - 2B cos(2 pi k / N)
        of the matrix K in S = x^T K x / 2.
    """
    A, B = coefficients(a, m, w)
    theta = 2 * np.pi * np.arange(N) / N
    lam = 2 * A - 2 * B * np.cos(theta)  # eigenvalues of K
    return np.array([np.mean(np.cos(theta * j) / lam) for j in range(N)])


def exact_x2(N, a, m, w):
    """
    Exact lattice <x^2> on the periodic lattice.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    float
        <x_t^2>, the same for every t.
    """
    A, B = coefficients(a, m, w)
    k = np.arange(N)
    return np.mean(1 / (2 * A - 2 * B * np.cos(2 * np.pi * k / N)))


def exact_values(N, a, m, w):
    """
    Exact lattice and continuum expectation values of the measured observables.

    The continuum values are for the oscillator at temperature 1/beta with
    beta = N a, which is what a periodic lattice of length N a describes.

    Parameters
    ----------
    N : int
        Number of time slices.
    a, m, w : float
        Lattice spacing, mass and frequency.

    Returns
    -------
    lattice : dict
        Exact lattice values of "x", "p", "V", "T_prim", "E_prim" and "E_vir".
    continuum : dict
        The same observables in the continuum at beta = N a.
    """
    C = exact_correlator(N, a, m, w)
    x2, dx2 = C[0], 2 * (C[0] - C[1])  # <x^2>, <(x_{t+1} - x_t)^2>
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
    return lattice, continuum
