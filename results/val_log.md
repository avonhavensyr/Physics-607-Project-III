# Validation run, 2026-11-14 23:48

Tests [6, 8, 11], Delta 2.5, seed 2026

Main chains: N=64, a=0.25, 4 chains (2 from x = 0, 2 from a wide random path), 8000 burn-in + 40000 sweeps each, Delta=2.5 (delta=1.231), acceptance 0.434, 0.434, 0.434, 0.434

| Test | Name | Result |
|---|---|---|
| 6 | Symmetries and starts | PASS |
| 8 | Acceptance and step size | PASS |
| 11 | Multi-walker test | PASS |

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
