# sho-mcmc

Metropolis sampling of the quantum harmonic oscillator path integral on a
periodic time lattice, with emcee as a reference sampler (Project 3).

## Install

From the repository folder:

    python3 -m pip install -e .

## Run

    sho-mcmc --N 64 --a 0.25 --seed 1 --out run1           # hand-written Metropolis
    sho-mcmc --N 64 --a 0.25 --seed 1 --emcee --out run1   # also emcee (slow)
    sho-mcmc-validate --out val1                           # Tests 1-8 and 11
    sho-mcmc-inject --out inj1                             # Test 10
    python3 scripts/break_code.py                          # Test 12: bugs on purpose (6 min)
    python3 -m sho_mcmc.multiwalker                        # Alexandra's multi-walker check
    sho-mcmc --help

If your terminal cannot find the commands, use `python3 -m sho_mcmc`,
`python3 -m sho_mcmc.validate` and `python3 -m sho_mcmc.injection` instead.
With `--out run1` the figures are saved together with a file holding the
settings, the seed and the results, so every run can be repeated.

## Reproduce every result

    bash scripts/run_all.sh

This writes everything into `results/` (about 12 minutes).

## Layout

    pyproject.toml          package metadata, dependencies and commands
    src/sho_mcmc/
        lattice.py          lattice action and exact results
        metropolis.py       hand-written checkerboard Metropolis sampler
        ensemble.py         emcee on the same target (reference sampler)
        diagnostics.py      binning errors and autocorrelation times
        chains.py           several chains, error bars and z-scores
        plots.py            summary and diagnostic figures
        verify.py           physics and symmetry checks, Metropolis vs emcee
        main.py             the sho-mcmc command
        checks.py           Tests 1-5: action, covariance, error bars, small lattice, known answers
        symmetries.py       Test 6: symmetries and starting points
        convergence.py      Test 7: R-hat and effective sample size
        tuning.py           Test 8: acceptance and choice of step size
        injection.py        Test 10: injection and recovery (sho-mcmc-inject)
        walkers.py          Test 11: multi-walker test
        multiwalker.py      Alexandra's multi-walker check
        validate.py         the sho-mcmc-validate command
    scripts/run_all.sh      reproduces everything in results/
    scripts/break_code.py   Test 12: puts bugs into a copy of the code
    results/                outputs of the runs used in the report
    Validation_Final.pdf    the validation plan: original
    Validation_All_Versions.tex     Validation revisions
    Report.pdf  the report
