"""Error bars and autocorrelation times for correlated Monte Carlo series."""

import numpy as np


def binning_error(series, min_blocks=32):
    """
    Error of the mean of a correlated time series by binning.

    Parameters
    ----------
    series : array_like
        One measurement per sweep.
    min_blocks : int, optional
        Block sizes 1, 2, 4, ... are used while at least this many blocks
        remain.

    Returns
    -------
    sizes : ndarray
        Block sizes.
    errs : ndarray
        Error of the mean for each block size. The value at the largest block
        size is the estimate, valid once the curve has reached a plateau.
    errs_err : ndarray
        Statistical uncertainty of each error estimate.
    """
    series = np.asarray(series)
    sizes, errs, errs_err = [], [], []
    b = 1
    while len(series) // b >= min_blocks:
        nb = len(series) // b
        blocks = series[: nb * b].reshape(nb, b).mean(1)
        e = blocks.std(ddof=1) / np.sqrt(nb)
        sizes.append(b)
        errs.append(e)
        errs_err.append(e / np.sqrt(2 * (nb - 1)))
        b *= 2
    return np.array(sizes), np.array(errs), np.array(errs_err)


def autocorrelation(series, c=6.0):
    """
    Normalized autocorrelation function and integrated autocorrelation time.

    tau_int = 1/2 + sum_t rho(t) is summed up to the first window W with
    W >= c tau_int(W) (Sokal's automatic window). The convention matches the
    binning estimate: err^2 = 2 tau_int var / n.

    Parameters
    ----------
    series : array_like
        One measurement per sweep.
    c : float, optional
        Window constant.

    Returns
    -------
    rho : ndarray
        Normalized autocorrelation rho(t) for t = 0, ..., n-1.
    tau : float
        Integrated autocorrelation time in sweeps.
    W : int
        Window used for the sum.
    """
    y = np.asarray(series) - np.mean(series)
    n = len(y)
    f = np.fft.rfft(y, 2 * n)  # zero padding avoids wrap-around
    acf = np.fft.irfft(f * np.conj(f))[:n]
    rho = acf / acf[0]
    taus = 0.5 + np.cumsum(rho[1:])
    W = next((t for t in range(1, n) if t >= c * taus[t - 1]), n - 1)
    return rho, taus[W - 1], W
