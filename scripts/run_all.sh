#!/usr/bin/env bash
# Reproduce every result in results/. Run from the repository folder:
#     bash scripts/run_all.sh
# Takes about 12 minutes; the emcee part takes 1-2 minutes and Test 12 about 6.
set -e
mkdir -p results
cd results

# Main run on the plan's main lattice (N = 64, a = 0.25, 8,000 + 40,000 sweeps)
# with the step chosen in Test 8 (Delta = 2.5, i.e. delta = 2.5 / sqrt(A) = 1.231),
# hand-written Metropolis against emcee: physics and symmetry checks (Kevin), Test 9.
python3 -m sho_mcmc --N 64 --a 0.25 --sweeps 48000 --burn 8000 --delta 1.231 \
    --seed 2026 --emcee --out main --no-show

# Tests 1-8 and 11 of the validation plan
python3 -m sho_mcmc.validate --out val --no-show

# Test 10: injection and recovery of m and w, with the negative control
python3 -m sho_mcmc.injection --out inj --no-show

# Test 12: put bugs into a copy of the code on purpose; every bug must be caught
python3 ../scripts/break_code.py | tee break_code.txt

echo "done: everything is in results/"
