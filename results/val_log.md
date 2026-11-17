# Validation run, 2026-11-16 19:26

Tests [1, 2, 3, 4, 5, 6, 7, 8, 11], Delta 2.5, seed 2026

Main chains: N=64, a=0.25, 4 chains (2 from x = 0, 2 from a wide random path), 8000 burn-in + 40000 sweeps each, Delta=2.5 (delta=1.231), acceptance 0.434, 0.434, 0.434, 0.434

| Test | Name | Result |
|---|---|---|
| 1 | Action and update | PASS |
| 2 | Exact covariance | PASS |
| 3 | Error-bar code | PASS |
| 4 | Smallest lattice (N = 2) | PASS |
| 5 | Known answers (N = 64) | PASS |
| 6 | Symmetries and starts | PASS |
| 7 | Convergence (R-hat, ESS, traces) | PASS |
| 8 | Acceptance and step size | PASS |
| 11 | Multi-walker test | PASS |

## Test 1: Action and update (PASS)

```
1000 random paths: max |dS - (S_new - S_old)| = 1.3e-13 (limit 1e-10)
1000 random paths: max |x^T K x / 2 - S(x)| = 1.1e-13 (limit 1e-10)
acceptance probability always between 0 and 1: True
checkerboard sweep vs site-by-site loop (same random numbers): max difference 0 (must be exactly 0)
```

## Test 2: Exact covariance (PASS)

```
N=64, a=0.25: K symmetric: True; smallest eigenvalue 0.2500 > 0: True
eigenvalues vs 2A - 2B cos(2 pi k/N): max difference 8.9e-15
C(j) from NumPy's inverse vs eigenvalue sum: 2.5e-15; vs closed form: 2.2e-16
values: C(0) = 0.4961, C(1) = 0.3866, C(2) = 0.3013, C(4) = 0.1830, C(8) = 0.0675
```

## Test 3: Error-bar code (PASS)

```
phi=0.5: tau_int    1.50 vs exact   1.50 (+0.3%, limit 10%)  error (blocks of 32) 8.30e-04 vs exact 8.46e-04 (-1.9%, limit 20%)
phi=0.9: tau_int    9.45 vs exact   9.50 (-0.6%, limit 10%)  error (blocks of 256) 2.11e-03 vs exact 2.13e-03 (-1.0%, limit 20%)
phi=0.99: tau_int   98.19 vs exact  99.50 (-1.3%, limit 10%)  error (blocks of 2048) 6.55e-03 vs exact 6.89e-03 (-5.0%, limit 20%)
R-hat: 4 matching chains 1.0001 (must be < 1.01); one chain shifted 1.0232 (must be > 1.01)
```

## Test 4: Smallest lattice (N = 2) (PASS)

```
KS test vs exact Gaussian (every 52th sweep, 3827 draws): x0+x1 p = 0.716, x0-x1 p = 0.409 (limit 0.001)
var(x0+x1)                    2.0000 ± 0.0214   expected   2.0000    +0.0σ  ok
var(x0-x1)                    0.3990 ± 0.0022   expected   0.4000    -0.5σ  ok
<(x0+x1)^4>                  11.9883 ± 0.2911   expected  12.0000    -0.0σ  ok
<(x0-x1)^4>                   0.4797 ± 0.0060   expected   0.4800    -0.0σ  ok
<(x0+x1)(x0-x1)>             -0.0010 ± 0.0044   expected   0.0000    -0.2σ  ok
acceptance                    0.6732 ± 0.0007   expected   0.6746    -2.1σ  ok
```

## Test 5: Known answers (N = 64) (PASS)

```
<x>                          -0.0090 ± 0.0068   expected   0.0000    -1.3σ  ok
<x^2> = C(0)                  0.4964 ± 0.0028   expected   0.4961    +0.1σ  ok
C(1)                          0.3869 ± 0.0028   expected   0.3866    +0.1σ  ok
C(2)                          0.3016 ± 0.0027   expected   0.3013    +0.1σ  ok
C(4)                          0.1835 ± 0.0026   expected   0.1830    +0.2σ  ok
C(8)                          0.0685 ± 0.0022   expected   0.0675    +0.4σ  ok
<x^4> = 3 C(0)^2              0.7393 ± 0.0087   expected   0.7385    +0.1σ  ok
<V>                           0.2482 ± 0.0014   expected   0.2481    +0.1σ  ok
<T> (primitive)               0.2490 ± 0.0019   expected   0.2481    +0.5σ  ok
E (primitive)                 0.4972 ± 0.0020   expected   0.4961    +0.5σ  ok
E (virial)                    0.4964 ± 0.0028   expected   0.4961    +0.1σ  ok
```

## Test 6: Symmetries and starts (PASS)

```
<x_16>                       -0.0212 ± 0.0141   expected   0.0000    -1.5σ  ok
<x_32>                       -0.0173 ± 0.0137   expected   0.0000    -1.3σ  ok
<x_48>                       -0.0019 ± 0.0133   expected   0.0000    -0.1σ  ok
<x_32x_36> - <x_32x_28>      -0.0042 ± 0.0071   expected   0.0000    -0.6σ  ok
<x_0^2>                       0.4973 ± 0.0080   expected   0.4961    +0.1σ  ok
<x_16^2>                      0.4955 ± 0.0082   expected   0.4961    -0.1σ  ok
<x_32^2>                      0.4987 ± 0.0080   expected   0.4961    +0.3σ  ok
<x_48^2>                      0.5142 ± 0.0091   expected   0.4961    +2.0σ  ok
<x^2>: zero start 0.4946 ± 0.0041, wide start 0.4982 ± 0.0039: -0.6σ (limit 3)  ok
C(4): zero start 0.1827 ± 0.0038, wide start 0.1843 ± 0.0035: -0.3σ (limit 3)  ok
```

## Test 7: Convergence (R-hat, ESS, traces) (PASS)

```
x       R-hat 1.0037 (limit 1.01)   ESS    1342 (limit 400)   ok
x2      R-hat 1.0023 (limit 1.01)   ESS    3587 (limit 400)   ok
C1      R-hat 1.0024 (limit 1.01)   ESS    3494 (limit 400)   ok
C2      R-hat 1.0025 (limit 1.01)   ESS    3352 (limit 400)   ok
C4      R-hat 1.0025 (limit 1.01)   ESS    3211 (limit 400)   ok
C8      R-hat 1.0020 (limit 1.01)   ESS    3300 (limit 400)   ok
x4      R-hat 1.0023 (limit 1.01)   ESS    4124 (limit 400)   ok
x_mid   R-hat 1.0018 (limit 1.01)   ESS    2631 (limit 400)   ok
x2_mid  R-hat 1.0007 (limit 1.01)   ESS    7630 (limit 400)   ok
```

## Test 8: Acceptance and step size (PASS)

```
a=0.25 Delta=0.5 (delta=0.246): acceptance 0.8603 ± 0.0002, Eq. (8) 0.8604,  -0.6σ  ok
a=0.25 Delta=1.0 (delta=0.492): acceptance 0.7293 ± 0.0003, Eq. (8) 0.7291,  +0.5σ  ok
a=0.25 Delta=1.5 (delta=0.739): acceptance 0.6119 ± 0.0003, Eq. (8) 0.6125,  -2.1σ  ok
a=0.25 Delta=2.0 (delta=0.985): acceptance 0.5139 ± 0.0003, Eq. (8) 0.5139,  -0.1σ  ok
a=0.25 Delta=2.5 (delta=1.231): acceptance 0.4336 ± 0.0003, Eq. (8) 0.4338,  -0.8σ  ok
a=0.25 Delta=3.0 (delta=1.477): acceptance 0.3707 ± 0.0004, Eq. (8) 0.3704,  +1.0σ  ok
a=0.25 Delta=4.0 (delta=1.969): acceptance 0.2814 ± 0.0003, Eq. (8) 0.2816,  -0.6σ  ok
a=1.0  Delta=0.5 (delta=0.408): acceptance 0.8577 ± 0.0015, Eq. (8) 0.8604,  -1.7σ  ok
a=1.0  Delta=1.0 (delta=0.816): acceptance 0.7285 ± 0.0017, Eq. (8) 0.7291,  -0.4σ  ok
a=1.0  Delta=1.5 (delta=1.225): acceptance 0.6091 ± 0.0019, Eq. (8) 0.6125,  -1.7σ  ok
a=1.0  Delta=2.0 (delta=1.633): acceptance 0.5139 ± 0.0017, Eq. (8) 0.5139,  -0.0σ  ok
a=1.0  Delta=2.5 (delta=2.041): acceptance 0.4315 ± 0.0018, Eq. (8) 0.4338,  -1.3σ  ok
a=1.0  Delta=3.0 (delta=2.449): acceptance 0.3697 ± 0.0018, Eq. (8) 0.3704,  -0.4σ  ok
a=1.0  Delta=4.0 (delta=3.266): acceptance 0.2855 ± 0.0019, Eq. (8) 0.2816,  +2.0σ  ok
step-size study at a = 0.25 (slowest of path average and x^2):
  Delta=0.5  acceptance 0.860  tau_int  380.8 sweeps  effective samples per 1000 sweeps   1.3
  Delta=1.0  acceptance 0.729  tau_int  153.5 sweeps  effective samples per 1000 sweeps   3.3
  Delta=1.5  acceptance 0.612  tau_int   75.8 sweeps  effective samples per 1000 sweeps   6.6
  Delta=2.0  acceptance 0.514  tau_int   61.9 sweeps  effective samples per 1000 sweeps   8.1
  Delta=2.5  acceptance 0.434  tau_int   55.9 sweeps  effective samples per 1000 sweeps   8.9
  Delta=3.0  acceptance 0.371  tau_int   57.8 sweeps  effective samples per 1000 sweeps   8.6
  Delta=4.0  acceptance 0.281  tau_int   91.3 sweeps  effective samples per 1000 sweeps   5.5
best Delta with acceptance in (0.35, 0.65): 2.5 (the plan expected about 2.5)
```

## Test 11: Multi-walker test (PASS)

```
x_mid: interval |q| < 0.7044, exact fraction 0.6827
  walker 0 (zero start): 0.6757 ± 0.0111   -0.6σ  ok
  walker 1 (zero start): 0.6892 ± 0.0080   +0.8σ  ok
  walker 2 (zero start): 0.6763 ± 0.0100   -0.6σ  ok
  walker 3 (zero start): 0.6618 ± 0.0105   -2.0σ  ok
  walker 4 (wide start): 0.6883 ± 0.0089   +0.6σ  ok
  walker 5 (wide start): 0.6919 ± 0.0073   +1.3σ  ok
  walker 6 (wide start): 0.6891 ± 0.0086   +0.7σ  ok
  walker 7 (wide start): 0.6956 ± 0.0090   +1.4σ  ok
  walkers agree: chi2 = 9.2 / 7, p = 0.238 (limit 0.001)  ok
path average: interval |q| < 0.2500, exact fraction 0.6827
  walker 0 (zero start): 0.6917 ± 0.0129   +0.7σ  ok
  walker 1 (zero start): 0.6739 ± 0.0160   -0.5σ  ok
  walker 2 (zero start): 0.6886 ± 0.0147   +0.4σ  ok
  walker 3 (zero start): 0.6774 ± 0.0171   -0.3σ  ok
  walker 4 (wide start): 0.7025 ± 0.0125   +1.6σ  ok
  walker 5 (wide start): 0.6751 ± 0.0128   -0.6σ  ok
  walker 6 (wide start): 0.6899 ± 0.0127   +0.6σ  ok
  walker 7 (wide start): 0.6839 ± 0.0150   +0.1σ  ok
  walkers agree: chi2 = 3.6 / 7, p = 0.823 (limit 0.001)  ok
```
