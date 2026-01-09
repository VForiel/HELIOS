---
title: MMI Usage Guide
---

# MMI Usage Guide

This page shows how to run MMI simulations and calibrations in HELIOS.

## Basic simulation

```python
from helios.sim.mmi import simulate
import numpy as np

# 2×2 example
out = simulate(
    N=2, M=2,
    W=10e-6,           # slab width [m]
    L=None,            # if None, uses L_pi/2 heuristic
    n_core=2.0458,     # core index
    delta_n=0.0958,    # n_core - n_clad
    wavelength=1.55e-6,
    input_amplitudes=[1/np.sqrt(2), 1/np.sqrt(2)],
    Din=None, Dout=None,   # default symmetric spacing
    Sin=None, Sout=None,   # default port widths
    num_modes=50,          # transverse modes in MMI
    z_resolution=None,     # use default
    verbose=False,
)

intensities = np.abs(out)**2
print("Output intensities:", intensities)
```

Notes:

- If `L=None`, HELIOS computes an $L_\pi$‑based default (see Numerics) and typically uses $L_\pi/2$.
- `num_modes` is an upper bound. Physics determines the actual guided count via cutoff; asking for more does not create non‑physical modes.

## Phase calibration

```python
from helios.sim.mmi import calibrate_input_phases_genetic

res = calibrate_input_phases_genetic(
    N=4, M=4, W=10e-6,
    n_core=2.0458, delta_n=0.0958, wavelength=1.55e-6,
    bright_output_idx=0,      # index to maximize
    num_modes=50,
    verbose=False,
)

print("Best metric (null/bright):", res["best_metric"]) 
print("Best phases [rad]:", res["best_phases"]) 
```

## Joint calibration of n_core and phases

```python
from helios.sim.mmi import calibrate_n_core_and_phases

res = calibrate_n_core_and_phases(
    N=4, M=4, W=10e-6,
    n_core_initial=2.0458, delta_n=0.0958, wavelength=1.55e-6,
    n_core_steps_coarse=20,
    gradient_convergence_threshold=1e-3,
    verbose=False,
)

print("Best n_core:", res["best_n_core"]) 
print("Best metric:", res["best_metric"]) 
```

## Multimode outputs

- Set larger `Sout` to explore multimode output coupling.
- HELIOS computes the V‑number and distributes coupling among LP modes if multimode.

## Controlling warnings

During calibrations, LP‑mode warnings are automatically suppressed. To globally disable them:

```bash
# PowerShell
$env:HELIOS_SUPPRESS_LP_WARNINGS="1"

# Bash
export HELIOS_SUPPRESS_LP_WARNINGS=1
```
