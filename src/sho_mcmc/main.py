"""
Command-line entry point ``sho-mcmc``: run the simulation and the physics checks.

Examples::

    sho-mcmc --N 64 --a 0.25 --sweeps 50000 --burn 5000 --seed 1 --out run1
    sho-mcmc --N 64 --a 0.25 --seed 1 --emcee --out run1   # also emcee (slow)
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
        "on a periodic time lattice, with emcee as an optional reference.",
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
    p.add_argument(
        "--emcee", action="store_true", help="also run emcee as a reference (slow)"
    )
    p.add_argument("--walkers", type=int, default=None, help="emcee walkers (2N)")
    p.add_argument("--steps", type=int, default=100_000, help="emcee steps (100000)")
    p.add_argument(
        "--emcee-thin", type=int, default=50, help="emcee: keep every n-th step (50)"
    )
    p.add_argument(
        "--emcee-burn", type=int, default=20_000, help="emcee burn-in steps (20000)"
    )
    p.add_argument(
        "--x0-spread", type=float, default=0.1, help="emcee start spread (0.1)"
    )
    p.add_argument("--out", default=None, help="prefix for saved figures and results")
    p.add_argument("--no-show", action="store_true", help="do not open plot windows")
    return p.parse_args(argv)


def main(argv=None):
    """
    Run the samplers, print the checks against exact results, and plot.

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

    runs = {"Metropolis": (chain, acc, args.burn, 1)}
    if args.emcee:
        from .ensemble import emcee_sampler  # emcee is only needed for --emcee

        walkers = args.walkers or 2 * args.N
        chain_e, acc_e = emcee_sampler(
            *params,
            walkers,
            args.steps,
            args.emcee_thin,
            args.x0,
            args.x0_spread,
            args.seed,
        )
        burn_rows = args.emcee_burn // args.emcee_thin
        runs["emcee"] = (chain_e, acc_e, burn_rows, args.emcee_thin)
    print()
    results = verify(runs, *params, save=out)

    if out:
        tables = {
            name: {k: {"mean": v[0], "err": v[1], "flag": v[2]} for k, v in res.items()}
            for name, res in results.items()
        }
        record = {
            "settings": vars(args),
            "acceptance": acc,
            "x2": {"mc": mc, "err": err, "lattice": lat, "continuum": cont},
            "results": tables,
        }
        with open(f"{out}_results.json", "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nsaved {out}_*.png and {out}_results.json")
    if not args.no_show:
        plt.show()
