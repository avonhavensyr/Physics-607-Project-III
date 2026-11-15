"""
Command-line entry point ``sho-mcmc``: run the Metropolis simulation.

Example::

    sho-mcmc --N 64 --a 0.25 --sweeps 50000 --burn 5000 --seed 1 --out run1
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from .metropolis import metropolis
from .plots import plot_diagnostics, summary
from .verify import verify


def parse_args(argv=None):
    """
    Read the command-line options.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; default is ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The options.
    """
    p = argparse.ArgumentParser(
        prog="sho-mcmc",
        description="Metropolis sampling of the harmonic oscillator path integral "
        "on a periodic time lattice.",
    )
    p.add_argument("--N", type=int, default=64, help="time slices, even (default 64)")
    p.add_argument("--a", type=float, default=0.25, help="lattice spacing (0.25)")
    p.add_argument("--m", type=float, default=1.0, help="mass (1)")
    p.add_argument("--w", type=float, default=1.0, help="frequency (1)")
    p.add_argument("--sweeps", type=int, default=50_000, help="sweeps (50000)")
    p.add_argument("--burn", type=int, default=5_000, help="burn-in sweeps (5000)")
    p.add_argument("--delta", type=float, default=1.0, help="proposal half-width (1)")
    p.add_argument("--x0", type=float, default=0.0, help="starting value of all x_t")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: new)")
    p.add_argument("--thin", type=int, default=10, help="thinning for the histogram")
    p.add_argument("--out", default=None, help="prefix for saved figures and results")
    p.add_argument("--no-show", action="store_true", help="do not open plot windows")
    return p.parse_args(argv)


def main(argv=None):
    """
    Run the simulation, print the checks against exact results, and plot.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default is ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    if args.seed is None:
        # pick a seed and report it, so that every run can be repeated
        args.seed = int(np.random.SeedSequence().entropy % 2**32)
    print(f"seed = {args.seed}")

    chain, acc = metropolis(
        args.N, args.a, args.m, args.w, args.sweeps, args.delta, args.x0, args.seed
    )
    out = args.out
    params = (args.N, args.a, args.m, args.w)
    bin_file = f"{out}_binning.png" if out else None
    mc, err, lat, cont = summary(chain, acc, *params, args.burn, filename=bin_file)
    diag_file = f"{out}_diagnostics.png" if out else None
    plot_diagnostics(
        chain, args.a, args.m, args.w, args.burn, args.thin, filename=diag_file
    )
    print()
    results = verify(chain, acc, *params, args.burn, save=out)

    if out:
        observables = {k: {"mean": v[0], "err": v[1]} for k, v in results.items()}
        record = {
            "settings": vars(args),
            "acceptance": acc,
            "x2": {"mc": mc, "err": err, "lattice": lat, "continuum": cont},
            "observables": observables,
        }
        with open(f"{out}_results.json", "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nsaved {out}_*.png and {out}_results.json")
    if not args.no_show:
        plt.show()
