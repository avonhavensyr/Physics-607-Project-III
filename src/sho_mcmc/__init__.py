"""
Metropolis sampling of the harmonic oscillator path integral on a periodic lattice.

Modules
-------
lattice
    Lattice action and exact results.
metropolis
    Hand-written checkerboard Metropolis sampler.
ensemble
    emcee on the same target, the reference sampler (needs emcee).
diagnostics
    Error bars (binning) and autocorrelation times.
chains
    Several chains, error bars and z-scores.
plots
    Summary and diagnostic figures.
verify
    Physics and symmetry checks, Metropolis against emcee.
main
    Command-line entry point ``sho-mcmc``.
symmetries, tuning, injection, walkers
    Tests 6, 8, 10 and 11 of the validation plan.
validate
    Command-line entry point ``sho-mcmc-validate``.
"""

from .diagnostics import autocorrelation, binning_error
from .lattice import action, coefficients, exact_correlator, exact_values, exact_x2
from .metropolis import metropolis

__all__ = [
    "action",
    "autocorrelation",
    "binning_error",
    "coefficients",
    "exact_correlator",
    "exact_values",
    "exact_x2",
    "metropolis",
]
__version__ = "0.1.0"
