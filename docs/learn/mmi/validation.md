---
title: Validation & Diagnostics
---

# Validation & Diagnostics

Recommended checks to ensure your MMI setup is physically and numerically sound.

## Power conservation (normalized grids)

- Track $\int |E(x,z)|^2\,dx$ along $z$. Small drifts may occur due to windowing/visualization outside the core; the power inside the core should remain stable absent explicit loss models.

## Self‑imaging distances

- Verify that qualitative images (single or multiple) appear near the designed fractions of $L_\pi$.
- Adjust $L$ slightly to compensate for dispersion and fabrication offsets.

## Mode cutoff and stability

- Increasing `num_modes` beyond the physically guided count should not change results (extra modes are beyond cutoff and contribute evanescent tails only).
- If you observe sensitivity to `num_modes`, re‑examine width `W`, wavelength, and index contrast (you may be near marginal guidance for higher modes).

## Output coupling sanity checks

- Single‑mode outputs: coupling should be dominated by LP$_{01}$.
- Multimode outputs: coupling fractions should sum to the total overlap power; LP cutoff ordering should match the V‑number regime.

## Grid and resolution

- Increase transverse grid density if you see aliasing of high‑order fringes.
- The default `z_resolution` is typically adequate because propagation is analytical via $e^{i\beta_m z}$; visualization step density can be reduced for speed.
