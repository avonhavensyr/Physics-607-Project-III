# sho-mcmc

Metropolis sampling of the quantum harmonic oscillator path integral on a
periodic time lattice (Project 3).

## Install

From the repository folder:

    pip install -e .

## Run

    sho-mcmc --N 64 --a 0.25 --sweeps 50000 --burn 5000 --seed 1 --out run1
    sho-mcmc --help
    sho-mcmc-inject --out inj1

`sho-mcmc` runs the simulation, prints the checks against the exact results and
shows the figures. With `--out run1` it also saves `run1_*.png` and
`run1_results.json` (settings, seed and results), so every run can be repeated.
`sho-mcmc-inject` runs the injection-recovery test (validation plan, Test 10).

If your terminal cannot find the commands, use `python -m sho_mcmc` instead of
`sho-mcmc` and `python -m sho_mcmc.injection` instead of `sho-mcmc-inject`.
