
"""
Module for spectrum generation and manipulation.
"""

import numpy as np
import astropy.units as u
import astropy.constants as const
from typing import Optional, Tuple

def modified_blackbody(wavelengths: Optional[u.Quantity], temperature: u.Quantity, beta: float = 1.0,
                       lambda0: u.Quantity = 100 * u.um, norm: Optional[float] = None) -> Tuple[u.Quantity, u.Quantity]:
    """
    Compute a modified blackbody spectrum B_lambda(λ, T) * (λ / lambda0)^{-beta}.

    Parameters
    ----------
    wavelengths : astropy.Quantity, optional
        Wavelength grid (length units). If None, a default grid is created.
    temperature : astropy.Quantity
        Temperature of the blackbody (K).
    beta : float, optional
        Spectral index modification (default 1.0). Use 0.0 for pure blackbody.
    lambda0 : astropy.Quantity, optional
        Reference wavelength for modification (default 100 um).
    norm : float, optional
        Normalization factor to multiply the final spectrum by.

    Returns
    -------
    tuple
        (wavelengths, sed) where sed is in W / (m^2 um sr).
    """
    # If wavelengths not provided, create a default grid
    if wavelengths is None:
        # Default grid: 0.1 um to 100 um
        wavelengths = np.logspace(np.log10((0.1 * u.um).to(u.m).value), np.log10((100 * u.um).to(u.m).value), 200) * u.m

    # Ensure proper units
    wavelengths = wavelengths.to(u.m)
    T = temperature.to(u.K)

    h = const.h
    c = const.c
    kB = const.k_B

    wl = wavelengths
    # Planck function per unit wavelength B_lambda(λ, T)
    # B_lambda(T) = (2hc^2 / lambda^5) * (1 / (exp(hc/lambda k T) - 1))
    
    val_check = (h * c) / (wl * kB * T)
    # Check for potential overflow or underflow if T is very small or wl very small
    # But usually astropy handles quantities well. Let's rely on standard formula.
    
    exponent = (h * c) / (wl * kB * T)
    # Avoid overflow warnings by using np.expm1 with values, though astropy quantities might be safer deconstructed sometimes
    # Converting to dimensionless for numpy func
    exponent_val = exponent.decompose().value
    
    # Clip exponent to avoid overflow in exp
    exponent_val = np.clip(exponent_val, -700, 700) 
    
    term2 = 1.0 / np.expm1(exponent_val)
    
    term1 = (2 * h * c ** 2) / (wl ** 5)
    
    B = term1 * term2
    
    # Ensure B has radiance units (per steradian)
    B = B * (1.0 / u.sr)

    # Modified emissivity logic: (lambda / lambda0)^-beta
    # For a pure blackbody star, beta should be 0.
    if beta != 0:
        emissivity = (wl / lambda0.to(u.m)) ** (-beta)
        sed = B * emissivity
    else:
        sed = B

    # Convert to standard units: W / (m^2 um sr)
    # W / m^2 / m / sr -> W / m^2 / um / sr implies * 1e-6 in denominator or * 1e6 value
    sed = sed.to(u.W / (u.m ** 2 * u.um * u.sr))

    if norm is not None:
        sed = sed * float(norm)

    return wavelengths.to(u.um), sed
