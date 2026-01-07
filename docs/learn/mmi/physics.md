---
title: Physical Principles of MMIs
---

# Physical Principles of MMIs

MMI couplers exploit self-imaging in a multimode slab waveguide: a field launched into a multimode region reproduces one or multiple images of the input after a specific propagation distance. This enables compact power splitters/combiners (e.g., 1×2, 2×2, 1×N, N×N).

## Modal picture and self‑imaging

Consider a rectangular multimode region of width $W$ (in $x$) with effective index $n_\text{core}$, surrounded by cladding of index $n_\text{clad}$. For a scalar model, the transverse field can be decomposed into orthonormal guided modes $\{\psi_m(x)\}$ with propagation constants $\{\beta_m\}$:

$$ E(x,z) = \sum_{m} a_m\, \psi_m(x) \, e^{i\beta_m z}. $$

Different modes accumulate different phases; at certain distances, the modal phase differences realign and the input field is re‑imaged (self‑imaging). For two dominant modes $m=0,1$, a useful length scale is the beat length:

$$ L_\pi = \frac{\pi}{\beta_0 - \beta_1}. $$

More generally, self‑images appear at rational fractions of $L_\pi$ depending on symmetry and input excitation. In common MMI couplers, the practical design formula used in HELIOS is the parabolic‑approximation result:

$$ L_\pi \approx \frac{4\, n_\text{eff}\, W^2}{3\, \lambda}, $$

where $n_\text{eff}$ is an effective index for the MMI slab and $\lambda$ is the wavelength in vacuum. This approximation matches well for high‑index‑contrast integrated photonics designs and is used in code to set initial lengths.

## Single and multiple images

- Single self‑image (replica of the input): typically at $\tfrac{1}{2}L_\pi$ for symmetric excitations.
- Twofold (or N‑fold) images: occur at fractional lengths $L = L_\pi/N$ with parity factors depending on input symmetry. These enable 1→N splitters by placing outputs at the image locations.

## Guided vs evanescent behavior (cutoff)

Inside the core, a simple slab picture gives (heuristically)

$$ \beta_m^2 \approx (k_0 n_\text{core})^2 - k_{x,m}^2, \quad k_0 = \frac{2\pi}{\lambda}, $$

with transverse wavenumbers $k_{x,m}$ increasing with $m$ (denser oscillations across the width). When the square‑root becomes imaginary, the corresponding solution is not guided in the core and decays evanescently in the cladding. This naturally limits the number of physically guided modes, independent of how many modes you ask the solver to include.

## Output coupling and multimode fibers/waveguides

For output waveguides, the standard fiber optics V‑number governs how many LP modes are guided:

$$ V = \frac{\pi a}{\lambda} \sqrt{n_\text{core}^2 - n_\text{clad}^2}, \quad a = \tfrac{1}{2} d_\text{core}. $$

- $V < 2.405$: single‑mode (LP$_{01}$ only).
- $V > 2.405$: multimode (LP$_{11}$, LP$_{21}$, LP$_{02}$, … progressively guided).

Coupling from the MMI field to the output waveguide is computed via overlap integrals $\eta \propto |\int E_\text{MMI}(x)\, \psi_{\text{out}}^*(x)\, dx|^2$. In multimode outputs, energy splits across all guided LP modes.
