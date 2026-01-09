---
title: Numerical Implementation in HELIOS
---

# Numerical Implementation in HELIOS

This page connects the physics to the HELIOS implementation (`helios.sim.mmi`). We outline the discrete model, mode handling, propagation, and output coupling.

## Field representation and propagation

We discretize a transverse grid $x$ over $[-W/2, W/2]$ (with evanescent padding outside for visualization) and propagate the field $E(x,z)$ through the MMI length $L$.

Modal decomposition (scalar slab model):

$$ E(x,z) = \sum_m a_m\, \psi_m(x)\, e^{i\beta_m z}, \quad \beta_m^2 \approx (k_0 n_\text{core})^2 - k_{x,m}^2, \quad k_0=\tfrac{2\pi}{\lambda}. $$

Key points used in HELIOS:

- Transverse wavenumbers $k_{x,m}$ grow with $m$; higher $m$ oscillate faster across the width.
- When $\beta_m$ becomes imaginary (negative radicand), the mode is beyond cutoff → treated as evanescent (no guided power contribution).
- Outside the core, evanescent tails decay with $\kappa = \sqrt{k_{x,m}^2 - (k_0 n_\text{clad})^2}$ and are shown for completeness.

## Self‑imaging length and defaults

The code uses the practical approximation

$$ L_\pi \approx \frac{4\, n_\text{eff}\, W^2}{3\, \lambda} $$

to set initial/fallback MMI lengths. Here $n_\text{eff}$ is a weighted average between core and cladding indices for the fundamental behavior. You can override $L$ explicitly.

## Inputs and calibration

- Inputs are Gaussian‑like fundamental modes positioned at the input ports. Their complex weights are optimized for routing (e.g., bright vs null outputs).
- Phase calibration: a deterministic coordinate‑descent (genetic‑like) loop minimizes a null‑depth metric.
- Joint calibration of $n_\text{core}$ and phases: coarse scan over $n_\text{core}$ followed by gradient refinement. This is robust to poor initial guesses.

## Output coupling: single‑ vs multi‑mode

- Single‑mode case: overlap integral with a Gaussian approximation of LP$_{01}$.
- Multimode case: compute V‑number and, if multimode, distribute coupling via overlap to the guided LP modes (LP$_{01}$, LP$_{11}$, LP$_{21}$, LP$_{02}$, …) up to a small max (configurable). Total coupling is the sum of per‑mode overlap powers.

Important formula (V‑number):

$$ V = \frac{\pi a}{\lambda} \sqrt{n_\text{core}^2 - n_\text{clad}^2}, \quad a = \tfrac{1}{2} d_\text{core}. $$

LP cutoffs (first ones):

- LP$_{01}$: $V_\text{cut}=0$, always guided;
- LP$_{11}$: $V_\text{cut}\approx 2.405$; LP$_{21}$ and LP$_{02}$: $\approx 3.832$; etc.

## Performance and stability

- Vectorized NumPy operations; analytical propagation; no per‑step FFT loops.
- Auto‑guards against non‑physical settings (ports outside width, negative spacings, division by zero normalizations).
- Warning control: noisy LP‑mode warnings are suppressed during calibration/\(n_\text{core}\) estimation and can be globally disabled via `HELIOS_SUPPRESS_LP_WARNINGS=1`.
