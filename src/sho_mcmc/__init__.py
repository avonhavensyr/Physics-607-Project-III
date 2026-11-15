"""
Metropolis sampling of the harmonic oscillator path integral on a periodic lattice.

Modules
-------
lattice
    Lattice action and exact results.
metropolis
    Hand-written checkerboard Metropolis sampler.
diagnostics
    Error bars (binning) and autocorrelation times.
plots
    Summary and diagnostic figures.
verify
    Physics checks against exact lattice and continuum values.
injection
    Injection-recovery test (validation plan, Test 10).
main
    Command-line entry point ``sho-mcmc``.
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
