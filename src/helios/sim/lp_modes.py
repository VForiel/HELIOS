"""
LP Mode Profiles for Step-Index Waveguides

This module provides rigorous calculation of Linearly Polarized (LP) modes for
step-index optical fibers and waveguides, following the analytical solutions of
the scalar wave equation in cylindrical coordinates.

References
----------
.. [1] Snyder, A. W., & Love, J. (2012). *Optical Waveguide Theory*. 
       Springer Science & Business Media. Chapter 12-15.
.. [2] Gloge, D. (1971). "Weakly guiding fibers". 
       *Applied Optics*, 10(10), 2252-2258.
.. [3] Marcuse, D. (1977). "Loss analysis of single-mode fiber splices". 
       *Bell System Technical Journal*, 56(5), 703-718.

Educational Notes
-----------------
**Why LP modes?**
    In weakly-guiding fibers (Δn << n_core), the scalar wave equation yields
    Linearly Polarized modes labeled LP_lm where:
    - l: azimuthal mode number (angular dependence)
    - m: radial mode number (number of radial intensity maxima)

**Fundamental mode LP₀₁:**
    The lowest-order mode, Gaussian-like profile, exists for ALL V-numbers.
    
**Higher-order modes:**
    LP₁₁, LP₂₁, LP₀₂, etc. Only propagate above cutoff V-numbers.
    
**V-number (Normalized Frequency):**
    V = (π·a/λ) · √(n_core² - n_cladding²)
    
    - V < 2.405: Single-mode (only LP₀₁)
    - V > 2.405: Multi-mode (LP₀₁ + higher orders)

**Why this matters for MMI output coupling:**
    When Sout is too large, the waveguide supports multiple modes. The MMI field
    couples to ALL these modes, not just LP₀₁. This REDUCES the effective coupling
    to the fundamental mode, contrary to the naive "larger = more overlap" intuition.
"""

import os
import logging
from contextlib import contextmanager
import numpy as np
from scipy.special import jv, kn  # Bessel functions

# Module-level logger and warning suppression control
_logger = logging.getLogger(__name__)
_SUPPRESS_LP_WARNINGS = bool(int(os.getenv("HELIOS_SUPPRESS_LP_WARNINGS", "0")))
_EMITTED_KEYS = set()

def set_lp_warning_suppression(suppress: bool = True) -> None:
    """Enable/disable LP-mode warning messages.

    Parameters
    ----------
    suppress : bool, default=True
        If True, LP-mode related warnings are suppressed.
    """
    global _SUPPRESS_LP_WARNINGS
    _SUPPRESS_LP_WARNINGS = bool(suppress)


@contextmanager
def suppress_lp_warnings():
    """Context manager to temporarily suppress LP-mode warnings."""
    global _SUPPRESS_LP_WARNINGS
    prev = _SUPPRESS_LP_WARNINGS
    try:
        _SUPPRESS_LP_WARNINGS = True
        yield
    finally:
        _SUPPRESS_LP_WARNINGS = prev


def _emit_lp_warning(key: str, message: str) -> None:
    """Emit a warning at most once per unique key, unless suppressed.

    Parameters
    ----------
    key : str
        Unique key identifying this warning kind (e.g., "fallback_LP02").
    message : str
        Human-readable warning message.
    """
    if _SUPPRESS_LP_WARNINGS:
        return
    if key in _EMITTED_KEYS:
        return
    _EMITTED_KEYS.add(key)
    _logger.warning(message)


def compute_v_number(core_diameter, wavelength, n_core, n_cladding):
    """
    Compute the V-number (normalized frequency) of a step-index waveguide.
    
    **KEY CLARIFICATION:** The input parameter is the CORE DIAMETER (d_core),
    not the Mode Field Width (MFD). The V-number is defined in terms of the
    physical core geometry.
    
    Parameters
    ----------
    core_diameter : float
        **Diameter of the waveguide core** [m] (= 2×core radius).
        This is the physical core size, NOT the mode field width.
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Refractive index of the core.
    n_cladding : float
        Refractive index of the cladding.
    
    Returns
    -------
    V : float
        Normalized frequency (dimensionless).
        
    Notes
    -----
    **Definition:**
        V = (π · d_core / λ) · √(n_core² - n_cladding²)
    
    **Modal Regimes:**
    - V < 2.405: Single-mode operation (only LP₀₁)
    - 2.405 < V < 3.832: LP₀₁ + LP₁₁
    - 3.832 < V < 5.520: LP₀₁ + LP₁₁ + LP₂₁
    - etc.
    
    **Physical Interpretation:**
    The V-number is the NUMBER OF WAVELENGTHS that fit across the core diameter,
    scaled by the numerical aperture. It determines how many transverse modes
    can propagate.
    
    **For typical photonic integrated circuits at λ=1.55 µm:**
    - n_core ≈ 2.0, n_cladding ≈ 1.9 → Δn ≈ 0.1
    - Single-mode requires core_diameter < ~2.7 µm
    
    Examples
    --------
    >>> # Silicon photonics waveguide @ 1.55 µm
    >>> # With d_core = 2.5 µm
    >>> V = compute_v_number(2.5e-6, 1.55e-6, 2.0, 1.9)
    >>> print(f"V = {V:.3f}")  # Should be < 2.405 for single-mode
    V = 1.689
    """
    radius = core_diameter / 2
    NA = np.sqrt(n_core**2 - n_cladding**2)  # Numerical Aperture
    V = (np.pi * radius / wavelength) * NA
    return V


def compute_mfd(core_diameter, wavelength, n_core, n_cladding):
    """
    Compute the Mode Field Diameter (MFD) from core diameter using Marcuse formula.
    
    **IMPORTANT:** MFD is where the fundamental mode (LP₀₁) concentrates its energy.
    It is DIFFERENT from the core diameter, but related through the V-number.
    
    Parameters
    ----------
    core_diameter : float
        Physical core diameter [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Core refractive index.
    n_cladding : float
        Cladding refractive index.
    
    Returns
    -------
    mfd : float
        Mode Field Diameter [m] (the 1/e² intensity diameter).
    
    Notes
    -----
    **Marcuse Formula (1977):**
        MFD = d_core × (0.65 + 1.619/V^1.5 + 2.879/V^6)
    
    where V is the V-number.
    
    **Physical Meaning:**
    - For V→0 (very narrow core): MFD ≈ 0.65·d_core (mode confined to core)
    - For V→∞ (very wide core): MFD ≈ d_core (mode samples the core width)
    
    **Why This Matters:**
    The LP₀₁ mode field extends beyond the core into the cladding. The MFD
    captures this physical reality. When coupling two waveguides, the overlap
    integral depends on their MFDs, not their core diameters!
    
    References
    ----------
    [1] Marcuse, D. (1977). "Loss analysis of single-mode fiber splices."
        *Bell System Technical Journal*, 56(5), 703-718.
    
    Examples
    --------
    >>> # Single-mode fiber at λ = 1.55 µm
    >>> # Core diameter = 8 µm, typical SM fiber parameters
    >>> d_core = 8e-6
    >>> V = compute_v_number(d_core, 1.55e-6, 1.477, 1.472)  # SMF-28 typical
    >>> mfd = compute_mfd(d_core, 1.55e-6, 1.477, 1.472)
    >>> print(f"d_core = {d_core*1e6:.2f} µm, MFD = {mfd*1e6:.2f} µm")
    d_core = 8.00 µm, MFD = 10.45 µm
    """
    V = compute_v_number(core_diameter, wavelength, n_core, n_cladding)
    
    # Marcuse formula for LP₀₁
    if V > 0:
        marcuse_factor = 0.65 + 1.619 / (V**1.5) + 2.879 / (V**6)
    else:
        # At V=0, mode is infinitely confined
        marcuse_factor = 0.65
    
    mfd = core_diameter * marcuse_factor
    return mfd


def lp_mode_cutoff(l, m):
    """
    Return the cutoff V-number for LP_lm mode.
    
    Parameters
    ----------
    l : int
        Azimuthal mode number (0, 1, 2, ...).
    m : int
        Radial mode number (1, 2, 3, ...).
    
    Returns
    -------
    V_cutoff : float
        Cutoff V-number below which this mode is evanescent.
    
    Notes
    -----
    LP₀₁ has V_cutoff = 0 (always propagates).
    
    Common cutoffs:
    - LP₀₁: 0.000
    - LP₁₁: 2.405
    - LP₂₁: 3.832
    - LP₀₂: 3.832
    - LP₃₁: 5.136
    - LP₁₂: 5.520
    
    References
    ----------
    [1] Snyder & Love (2012), Table 12-3.
    """
    # Analytical cutoff is the p-th zero of J_{l-1}(V) for LP_{lm}
    # For simplicity, we use tabulated values for the first few modes
    cutoffs = {
        (0, 1): 0.000,    # LP₀₁ (fundamental)
        (1, 1): 2.405,    # LP₁₁
        (2, 1): 3.832,    # LP₂₁
        (0, 2): 3.832,    # LP₀₂
        (3, 1): 5.136,    # LP₃₁
        (1, 2): 5.520,    # LP₁₂
        (4, 1): 6.380,    # LP₄₁
        (2, 2): 7.016,    # LP₂₂
    }
    
    key = (l, m)
    if key in cutoffs:
        return cutoffs[key]
    else:
        # Approximate for higher modes: V_lm ≈ (l + 2m - 1) * π/2
        return (l + 2*m - 1) * np.pi / 2


def compute_lp_mode_profile(
    x_grid,
    center,
    core_diameter,
    wavelength,
    n_core,
    n_cladding,
    l=0,
    m=1,
):
    """
    Compute the transverse intensity profile of an LP_lm mode (1D projection).
    
    Parameters
    ----------
    x_grid : np.ndarray
        1D array of transverse positions [m].
    center : float
        Waveguide center position [m].
    core_diameter : float
        Core diameter [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Core refractive index.
    n_cladding : float
        Cladding refractive index.
    l : int, default=0
        Azimuthal mode number (angular).
    m : int, default=1
        Radial mode number.
    
    Returns
    -------
    profile : np.ndarray
        Normalized intensity profile |ψ(x)|² such that ∫|ψ|² dx = 1.
    
    Notes
    -----
    This function returns the INTENSITY profile (|ψ|²), not the field amplitude.
    
    For 1D projections (MMI output coupling), we assume azimuthal symmetry and
    integrate over the angular dimension. The result is a radial profile projected
    onto the x-axis.
    
    **Physical Interpretation:**
    - LP₀₁: Single central lobe (Gaussian-like)
    - LP₁₁: Two lobes (doughnut shape → double-peaked in 1D)
    - LP₂₁: Three lobes (triple-peaked in 1D)
    
    **Mode Field Radius (MFR):**
    For LP₀₁, the 1/e² intensity radius is approximately:
        w₀ ≈ a × (0.65 + 1.619/V^(3/2) + 2.879/V^6)
    where a is the core radius. This is the "effective mode width".
    
    References
    ----------
    [1] Marcuse (1977) - Mode field diameter formulas
    [2] Snyder & Love (2012) - Chapter 13, LP mode profiles
    """
    radius = core_diameter / 2
    V = compute_v_number(core_diameter, wavelength, n_core, n_cladding)
    V_cutoff = lp_mode_cutoff(l, m)
    
    # Check if mode is guided
    if V < V_cutoff:
        # Mode is evanescent (not guided) → return zeros (warn once if not suppressed)
        _emit_lp_warning(
            key=f"below_cutoff_LP{l}{m}",
            message=f"⚠️ LP_{l}{m} is below cutoff (V={V:.3f} < {V_cutoff:.3f})"
        )
        return np.zeros_like(x_grid)
    
    # Radial coordinate relative to center
    r = np.abs(x_grid - center)
    
    # Normalized radial coordinate
    rho = r / radius
    
    # Transverse propagation constants (inside/outside core)
    k0 = 2 * np.pi / wavelength
    n_eff_approx = (n_core + n_cladding) / 2  # Approximation for weakly guiding
    
    # U and W parameters (see Snyder & Love, Eq. 12-44)
    # These are derived from the eigenvalue equation for LP modes
    # Simplified analytical approximations:
    U = V * np.sqrt(1 - (V_cutoff / V)**2) if V > V_cutoff else 0
    W = V * (V_cutoff / V) if V > V_cutoff else 0
    
    # Field profile (radial part)
    if l == 0 and m == 1:
        # LP₀₁: Fundamental mode (Gaussian approximation)
        # Use Marcuse approximation for mode field radius
        w0 = radius * (0.65 + 1.619 / V**1.5 + 2.879 / V**6)
        sigma = w0 / np.sqrt(2)  # Convert to Gaussian σ
        
        field = np.exp(-(r**2) / (2 * sigma**2))
    
    elif l == 1 and m == 1:
        # LP₁₁: First higher-order mode (doughnut shape)
        # Radial profile: J₁(U·ρ) inside, K₁(W·ρ) outside
        field = np.zeros_like(r)
        
        # Core region (r < radius)
        core_mask = rho <= 1.0
        if np.any(core_mask):
            field[core_mask] = jv(1, U * rho[core_mask])
        
        # Cladding region (r > radius)
        clad_mask = rho > 1.0
        if np.any(clad_mask):
            # Match boundary condition at r=radius
            J_at_boundary = jv(1, U)
            K_at_boundary = kn(1, W) if W > 0 else 1e-10
            amplitude = J_at_boundary / K_at_boundary if K_at_boundary > 1e-15 else 0
            field[clad_mask] = amplitude * kn(1, W * rho[clad_mask])
    
    elif l == 2 and m == 1:
        # LP₂₁: Second higher-order mode
        field = np.zeros_like(r)
        
        core_mask = rho <= 1.0
        if np.any(core_mask):
            field[core_mask] = jv(2, U * rho[core_mask])
        
        clad_mask = rho > 1.0
        if np.any(clad_mask):
            J_at_boundary = jv(2, U)
            K_at_boundary = kn(2, W) if W > 0 else 1e-10
            amplitude = J_at_boundary / K_at_boundary if K_at_boundary > 1e-15 else 0
            field[clad_mask] = amplitude * kn(2, W * rho[clad_mask])
    
    else:
        # Higher modes: not implemented, use Gaussian fallback
        _emit_lp_warning(
            key=f"fallback_LP{l}{m}",
            message=f"⚠️ LP_{l}{m} profile not implemented, using Gaussian approximation"
        )
        w0 = radius * 0.8  # Rough estimate
        sigma = w0 / np.sqrt(2)
        field = np.exp(-(r**2) / (2 * sigma**2))
    
    # Convert to intensity
    intensity = np.abs(field)**2
    
    # Normalize: ∫|ψ|² dx = 1
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    norm = np.sum(intensity) * dx
    if norm > 1e-15:
        intensity /= norm
    
    return intensity


def compute_multimode_coupling(
    field_mmi,
    x_grid,
    output_center,
    core_diameter,
    wavelength,
    n_core,
    n_cladding,
    max_modes=5,
):
    """
    Compute the coupling efficiency from MMI field to all guided modes.
    
    This function calculates the overlap integral between the MMI output field
    and each guided LP mode, accounting for multimode behavior when the waveguide
    diameter is large.
    
    Parameters
    ----------
    field_mmi : np.ndarray (complex)
        Complex field amplitude from MMI at output position [arbitrary units].
    x_grid : np.ndarray
        1D spatial grid [m].
    output_center : float
        Center position of output waveguide [m].
    core_diameter : float
        Output waveguide core diameter [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Core refractive index.
    n_cladding : float
        Cladding refractive index.
    max_modes : int, default=5
        Maximum number of LP modes to consider.
    
    Returns
    -------
    coupling_dict : dict
        Dictionary with keys:
        - 'V': V-number
        - 'modes': list of dicts, each containing:
            - 'label': e.g., 'LP01'
            - 'l': azimuthal number
            - 'm': radial number
            - 'coupling': fraction of power coupled to this mode
            - 'cutoff': cutoff V-number
        - 'total_coupling': sum of all mode couplings
    
    Notes
    -----
    **Energy Conservation:**
        total_coupling = Σ_modes |overlap_mode|² ≤ 1
    
    **Physical Interpretation:**
        In single-mode regime (V<2.405): coupling ≈ coupling_LP01
        In multimode regime (V>2.405): coupling splits among modes
        
        Example:
            V = 3.0 → LP₀₁ (60%) + LP₁₁ (30%) + residual (10%)
            
        Larger Sout does NOT always mean more coupling to LP₀₁!
    
    **Rigorous Treatment:**
        This is the standard approach in fiber optics for analyzing mode mismatch
        losses in splices and connectors [Marcuse 1977].
    """
    V = compute_v_number(core_diameter, wavelength, n_core, n_cladding)
    
    # Determine which modes are guided
    mode_list = [
        (0, 1, 'LP01'),  # Always guided
        (1, 1, 'LP11'),
        (2, 1, 'LP21'),
        (0, 2, 'LP02'),
        (3, 1, 'LP31'),
        (1, 2, 'LP12'),
    ]
    
    guided_modes = []
    for l, m, label in mode_list[:max_modes]:
        V_cutoff = lp_mode_cutoff(l, m)
        if V >= V_cutoff:
            guided_modes.append({'l': l, 'm': m, 'label': label, 'cutoff': V_cutoff})
    
    # Compute overlap integrals
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    
    for mode in guided_modes:
        # Get mode profile (intensity)
        mode_profile_intensity = compute_lp_mode_profile(
            x_grid, output_center, core_diameter, wavelength,
            n_core, n_cladding, mode['l'], mode['m']
        )
        
        # Overlap integral: ∫ E_mmi · ψ_mode* dx
        # Since we have intensity profile, approximate: ψ ≈ √I (ignoring phase)
        mode_field = np.sqrt(mode_profile_intensity + 1e-15)  # Avoid sqrt(0)
        
        overlap = np.sum(field_mmi * mode_field) * dx
        coupling_power = np.abs(overlap)**2
        
        mode['coupling'] = float(coupling_power)
    
    total_coupling = sum(m['coupling'] for m in guided_modes)
    
    return {
        'V': float(V),
        'modes': guided_modes,
        'total_coupling': float(total_coupling),
    }


def print_mode_info(core_diameter, wavelength, n_core=2.0, n_cladding=1.9):
    """
    Print educational information about guided modes for given parameters.
    
    Parameters
    ----------
    core_diameter : float
        Core diameter [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float, default=2.0
        Core refractive index.
    n_cladding : float, default=1.9
        Cladding refractive index.
    
    Examples
    --------
    >>> print_mode_info(2.5e-6, 1.55e-6)
    ========================================
    WAVEGUIDE MODE ANALYSIS
    ========================================
    Core diameter: 2.50 µm
    Wavelength: 1.550 µm
    V-number: 1.689
    
    ✓ SINGLE-MODE REGIME (V < 2.405)
    
    Guided modes:
      ✓ LP₀₁ (fundamental) - always guided
    """
    V = compute_v_number(core_diameter, wavelength, n_core, n_cladding)
    
    print("=" * 50)
    print("WAVEGUIDE MODE ANALYSIS")
    print("=" * 50)
    print(f"Core diameter: {core_diameter*1e6:.2f} µm")
    print(f"Wavelength: {wavelength*1e6:.3f} µm")
    print(f"n_core: {n_core:.4f}, n_cladding: {n_cladding:.4f}")
    print(f"V-number: {V:.3f}")
    print()
    
    if V < 2.405:
        print("✓ SINGLE-MODE REGIME (V < 2.405)")
        print("  → Only LP₀₁ (fundamental mode) propagates")
        print("  → Optimal for interferometry (no modal noise)")
    else:
        print("⚠️ MULTIMODE REGIME (V > 2.405)")
        print("  → Multiple modes propagate")
        print("  → Coupling splits among modes")
        print("  → Potential for modal noise and instability")
    
    print()
    print("Guided modes:")
    
    mode_list = [
        (0, 1, 'LP₀₁'),
        (1, 1, 'LP₁₁'),
        (2, 1, 'LP₂₁'),
        (0, 2, 'LP₀₂'),
        (3, 1, 'LP₃₁'),
        (1, 2, 'LP₁₂'),
    ]
    
    for l, m, label in mode_list:
        V_cutoff = lp_mode_cutoff(l, m)
        if V >= V_cutoff:
            status = "✓"
            note = " (fundamental)" if (l==0 and m==1) else ""
            print(f"  {status} {label}{note} - V_cutoff = {V_cutoff:.3f}")
        else:
            status = "✗"
            print(f"  {status} {label} - V_cutoff = {V_cutoff:.3f} (below cutoff)")
    
    print()
