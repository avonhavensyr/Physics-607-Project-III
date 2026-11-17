"""
Command-line entry point ``sho-mcmc-validate``: run the validation tests.

It runs Tests 1-8 and 11 of the plan; Test 9 is ``sho-mcmc --emcee`` and
Test 10 is ``sho-mcmc-inject``. Every test prints its numbers and PASS or
FAIL. With ``--out run1`` the figures are saved and every result is written
to ``run1_log.md``, so the report can say exactly what each test showed.

To add a test, write a function that returns a dict with "test" (its number),
"name", "passed" (bool), "lines" (list of str to print) and, optionally,
"figures" (dict of matplotlib figures), and add it to ``runners`` in `main`.
`sho_mcmc.chains` has helpers to run several chains and to compare measured
values with exact ones.
"""

import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np

from . import checks, convergence, symmetries, tuning, walkers
from .chains import run_chains
from .lattice import coefficients

MAIN = (64, 0.25, 1.0, 1.0)  # main lattice (N, a, m, w), plan Section 4
N_BURN, N_KEEP = 8_000, 40_000  # sweeps per chain, plan Section 4
DEFAULT_TESTS = [1, 2, 3, 4, 5, 6, 7, 8, 11]


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
        prog="sho-mcmc-validate",
        description="Run tests of the validation plan.",
    )
    p.add_argument(
        "--tests", type=int, nargs="+", default=DEFAULT_TESTS, help="tests to run"
    )
    p.add_argument(
        "--Delta", type=float, default=2.5, help="main-chain step in plan units (2.5)"
    )
    p.add_argument("--seed", type=int, default=2026, help="base random seed (2026)")
    p.add_argument("--out", default=None, help="prefix for figures and the log")
    p.add_argument("--no-show", action="store_true", help="do not open plot windows")
    return p.parse_args(argv)


def write_log(path, args, results, header):
    """
    Write the results as a Markdown log for the report.

    Parameters
    ----------
    path : str
        File to write.
    args : argparse.Namespace
        The options of this run.
    results : list of dict
        Test results.
    header : list of str
        Lines describing the main chains.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w") as f:
        f.write(f"# Validation run, {now}\n\n")
        f.write(f"Tests {args.tests}, Delta {args.Delta}, seed {args.seed}\n\n")
        for line in header:
            f.write(f"{line}\n\n")
        f.write("| Test | Name | Result |\n|---|---|---|\n")
        for r in results:
            f.write(f"| {r['test']} | {r['name']} | {_verdict(r)} |\n")
        for r in results:
            f.write(f"\n## Test {r['test']}: {r['name']} ({_verdict(r)})\n\n```\n")
            f.write("\n".join(r["lines"]) + "\n```\n")


def _verdict(result):
    """
    PASS or FAIL for one test result.

    Parameters
    ----------
    result : dict
        Test result.

    Returns
    -------
    str
        "PASS" or "FAIL".
    """
    return "PASS" if result["passed"] else "FAIL"


def main(argv=None):
    """
    Run the selected tests, print the results, and save them with --out.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default is ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    seed = args.seed
    tests = sorted(set(args.tests))

    main_chains = starts = None
    header = []
    if {5, 6, 7} & set(tests):  # Tests 5-7 use the four main chains (plan Section 4)
        A, _ = coefficients(*MAIN[1:])
        delta = args.Delta / np.sqrt(A)  # the code's step for the plan's Delta
        main_chains, acc, starts = run_chains(*MAIN, N_BURN, N_KEEP, delta, seed + 500)
        header.append(
            f"Main chains: N={MAIN[0]}, a={MAIN[1]}, 4 chains (2 from x = 0, "
            f"2 from a wide random path), {N_BURN} burn-in + {N_KEEP} sweeps each, "
            f"Delta={args.Delta} (delta={delta:.3f}), "
            f"acceptance {', '.join(f'{r:.3f}' for r in acc)}"
        )
        print(header[0])

    runners = {
        1: lambda: checks.test1_action_update(seed=seed + 1),
        2: lambda: checks.test2_exact_covariance(*MAIN),
        3: lambda: checks.test3_error_bars(seed=seed + 3),
        4: lambda: checks.test4_small_lattice(seed=seed + 4),
        5: lambda: checks.test5_known_answers(main_chains, N_BURN, *MAIN),
        6: lambda: symmetries.test6_symmetries(main_chains, N_BURN, starts, *MAIN),
        7: lambda: convergence.test7_convergence(main_chains, N_BURN, starts),
        8: lambda: tuning.test8_acceptance(seed=seed + 8),
        11: lambda: walkers.test11_walkers(Delta=args.Delta, seed=seed + 11),
    }
    results = []
    for t in tests:
        if t not in runners:
            print(
                f"\n== Test {t}: not in this command "
                "(Test 9: sho-mcmc --emcee, Test 10: sho-mcmc-inject)"
            )
            continue
        r = runners[t]()
        results.append(r)
        print(f"\n== Test {t}: {r['name']}  [{_verdict(r)}]")
        for line in r["lines"]:
            print("   " + line)

    print("\nSummary: " + ", ".join(f"Test {r['test']} {_verdict(r)}" for r in results))
    if args.out:
        for r in results:
            for name, fig in r.get("figures", {}).items():
                fig.savefig(f"{args.out}_test{r['test']}_{name}.png", dpi=130)
        write_log(f"{args.out}_log.md", args, results, header)
        print(f"saved {args.out}_log.md and the figures")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
