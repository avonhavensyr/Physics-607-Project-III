"""
Test 12 of the validation plan: break the code on purpose.

For each bug in BUGS, this script copies src/sho_mcmc into a temporary folder,
changes one line there, runs sho-mcmc-validate (Tests 1-8 and 11) and
sho-mcmc-inject (Test 10) on the broken copy, and prints which tests failed.
The real code is never changed. Run it from the repository folder:

    python3 scripts/break_code.py          # all bugs
    python3 scripts/break_code.py 2 4      # only bugs 2 and 4

Each bug takes one to two minutes. Every bug must make at least one test fail.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "sho_mcmc"

# (description, file, original text, broken text)
BUGS = [
    (
        "rejected steps are kept (no accept/reject)",
        "metropolis.py",
        "x = x + shift * acc",
        "x = x + shift * half",
    ),
    (
        "wrong sign of the neighbour term in dS",
        "metropolis.py",
        "- B * (np.roll(x, 1) + np.roll(x, -1))",
        "+ B * (np.roll(x, 1) + np.roll(x, -1))",
    ),
    (
        "wrong neighbour: x_{t-1} used twice in dS",
        "metropolis.py",
        "np.roll(x, 1) + np.roll(x, -1)",
        "2 * np.roll(x, 1)",
    ),
    (
        "missing factor m in the potential",
        "lattice.py",
        "return m / a + a * m * w**2 / 2, m / a",
        "return m / a + a * w**2 / 2, m / a",
    ),
    (
        "no checkerboard: all sites updated at once",
        "metropolis.py",
        "acc = half & (np.log(rng.random(N)) < -dS)",
        "acc = np.log(rng.random(N)) < -dS",
    ),
    (
        "error bars ignore the autocorrelation (tau_int = 1/2)",
        "diagnostics.py",
        "return rho, taus[W - 1], W",
        "return rho, 0.5, W",
    ),
]


def run(module, env):
    """
    Run one command of the package on the broken copy.

    Parameters
    ----------
    module : str
        Module to run with ``python -m``.
    env : dict
        Environment, with PYTHONPATH pointing at the broken copy.

    Returns
    -------
    str
        Everything the command printed.
    """
    cmd = [sys.executable, "-m", module, "--no-show"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return out.stdout + out.stderr


def main():
    """Apply the bugs in turn and report which tests caught each one."""
    chosen = [int(k) for k in sys.argv[1:]] or range(1, len(BUGS) + 1)
    rows = []
    for k in chosen:
        bug, file, old, new = BUGS[k - 1]
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "sho_mcmc"
            shutil.copytree(SRC, pkg, ignore=shutil.ignore_patterns("__pycache__"))
            text = (pkg / file).read_text()
            if text.count(old) != 1:
                sys.exit(f"bug {k}: the line to change was not found in {file}")
            (pkg / file).write_text(text.replace(old, new))

            path = os.pathsep.join(p for p in (tmp, os.environ.get("PYTHONPATH")) if p)
            env = dict(os.environ, PYTHONPATH=path, MPLBACKEND="Agg")
            where = subprocess.run(
                [sys.executable, "-c", "import sho_mcmc; print(sho_mcmc.__file__)"],
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()
            if Path(tmp).resolve() not in Path(where).resolve().parents:
                sys.exit("the broken copy was not used; is PYTHONPATH overridden?")

            print(f"\nBug {k}: {bug}  ({file})", flush=True)
            caught = []
            out = run("sho_mcmc.validate", env)
            if "Summary:" in out:
                caught += [f"Test {t}" for t in re.findall(r"Test (\d+) FAIL", out)]
            else:
                last = out.strip().splitlines()[-1] if out.strip() else "no output"
                caught.append(f"sho-mcmc-validate crashed ({last})")
            out = run("sho_mcmc.injection", env)
            m = re.search(r"(\d+)/(\d+) runs pass", out)
            if m is None:
                last = out.strip().splitlines()[-1] if out.strip() else "no output"
                caught.append(f"sho-mcmc-inject crashed ({last})")
            elif m.group(1) != m.group(2):
                caught.append(f"Test 10 ({m.group(1)}/{m.group(2)} cases pass)")
            print("   caught by: " + (", ".join(caught) if caught else "NOTHING"))
            rows.append((k, bug, caught))

    print("\nSummary")
    for k, bug, caught in rows:
        print(f"  bug {k}: {bug}: {', '.join(caught) if caught else 'NOT CAUGHT'}")
    missed = [k for k, _, caught in rows if not caught]
    if missed:
        print(f"Test 12 FAIL: bugs {missed} were not caught by any test")
    else:
        print("Test 12 PASS: every bug was caught by at least one test")


if __name__ == "__main__":
    main()
