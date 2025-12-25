
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

def simulate_lit_planet(wavelengths: u.Quantity, 
                       star_spectrum: Tuple[u.Quantity, u.Quantity], 
                       semi_major_axis: u.Quantity, 
                       radius: u.Quantity, 
                       distance: u.Quantity, 
                       albedo: float, 
                       teff: u.Quantity) -> Tuple[u.Quantity, u.Quantity, u.Quantity]:
    """
    Simulates the total flux (Reflected + Thermal) of a planet.
    
    Parameters
    ----------
    wavelengths : astropy.Quantity
        Wavelength grid for the simulation.
    star_spectrum : tuple
        (wl_star, flux_star_at_1au) 
        Reference star spectrum (usually at 1 AU).
    semi_major_axis : astropy.Quantity
        Distance from Star to Planet.
    radius : astropy.Quantity
        Planet Radius.
    distance : astropy.Quantity
        Distance from Planet to Observer.
    albedo : float
        Geometric Albedo (scalar approximation for synthetic part).
    teff : astropy.Quantity
        Effective Temperature of Planet (for Thermal Emission).
        
    Returns
    -------
    tuple
        (total_flux, reflected_component, thermal_component)
        All in Jy (Spectral Flux Density)
    """
    
    # Unpack Star Spectrum
    wl_star, flux_star_ref = star_spectrum
    
    # Intepolate Star Flux to requested wavelengths
    # flux_star_ref usually in Jy (Spectral Flux Density)
    # We need to handle units carefully.
    
    # Ensure inputs are Quantity
    if not isinstance(wavelengths, u.Quantity): wavelengths = wavelengths * u.um
    
    wl_req_val = wavelengths.to(u.um).value
    wl_star_val = wl_star.to(u.um).value
    flux_star_val = flux_star_ref.to(u.Jy).value
    
    flux_star_interp_val = np.interp(wl_req_val, wl_star_val, flux_star_val, left=0, right=0)
    flux_star_interp = flux_star_interp_val * u.Jy
                                 
    # 1. Reflected Component
    # F_pl_refl = F_star_at_PE * Albedo * Phase * (R_pl / D_obs)^2
    # F_star_at_PE = F_star_1AU * (1AU / a)^2
    
    flux_star_at_planet = flux_star_interp * ((1.0*u.AU / semi_major_axis)**2).decompose()
    
    # Geometry factor for Observer
    # Assuming Phase=0 (Full) for standard SED
    geom_factor = ((radius / distance)**2).decompose()
    
    flux_reflected = flux_star_at_planet * albedo * geom_factor
    
    # 2. Thermal Component
    # B_lambda(T) * SolidAngle
    # SolidAngle = pi * (R / D)^2
    
    _, bb_surface = modified_blackbody(wavelengths, teff, beta=0) # Pure Blackbody
    
    solid_angle = (np.pi * (radius / distance)**2).decompose() * u.sr
    
    flux_thermal_surface = bb_surface * solid_angle
    
    # Convert BB to Jy
    flux_thermal = flux_thermal_surface.to(u.Jy, equivalencies=u.spectral_density(wavelengths))
    
    total_flux = flux_reflected + flux_thermal
    
    return total_flux, flux_reflected, flux_thermal
