---
title: MMI Design Rules
---

# MMI Design Rules

This page summarizes practical rules to size and place an MMI’s inputs/outputs, and to select a suitable length.

## Geometrical parameters

- Core width: $W$ (slab region that supports many modes)
- Length: $L$ (chosen near fractions of $L_\pi$)
- Input/output port count: $N$ inputs, $M$ outputs
- Port spacing: $D_\text{in}$ and $D_\text{out}$ (center‑to‑center)
- Port widths: $S_\text{in}$ and $S_\text{out}$ (physical core diameters at access waveguides)

## Length selection (rule of thumb)

Starting point from the parabolic‑index approximation used in HELIOS:

$$ L_\pi \approx \frac{4\, n_\text{eff}\, W^2}{3\, \lambda}. $$

Common working lengths:

- Single image (1×1 focusing): $L \approx \tfrac{1}{2} L_\pi$
- Balanced 1×2 splitter: $L \approx \tfrac{1}{2} L_\pi$ with symmetric ports
- General 1×N splitter: $L \approx \tfrac{1}{N} L_\pi$ (with parity/symmetry considerations)
- NxN multiport: choose $L$ to align N images with the N target ports

Fine tuning of $L$ accounts for real dispersion of $\{\beta_m\}$, fabrication deviations, and wavelength dependence.

## Port positioning

- Inputs are placed symmetrically around $x=0$ within $[-W/2,\, W/2]$.
- Outputs are placed at the expected image locations (symmetry‑matched to inputs).
- Spacings must keep ports within the MMI width (HELIOS enforces bounds and will raise clear errors otherwise).

## Single‑ vs multi‑mode outputs

- Single‑mode outputs (low $S_\text{out}$) maximize coupling to LP$_{01}$ → desirable for stable interferometry.
- Larger $S_\text{out}$ increases $V$ and makes outputs multimode → coupling splits among LP modes and may degrade performance.

## Polarization and dispersion notes

- The formulas above assume a scalar, effective‑index model. Polarization may alter $n_\text{eff}$ and modal dispersion.
- Wavelength scaling is approximately $\propto W^2/\lambda$; always validate across your band of interest.
