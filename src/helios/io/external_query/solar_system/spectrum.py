
import numpy as np
from astropy import units as u
from helios.sim.spectrum import modified_blackbody
from .constants import SOLAR_SYSTEM_DATA

def get_solar_spectrum(wavelengths=None):
    """
    Returns the solar spectral irradiance (Flux at 1 AU).
    Preferably returns ASTM E-490 standard.
    Falls back to Blackbody if data not found.
    """
    if wavelengths is None:
        wavelengths = np.logspace(np.log10(0.1), np.log10(30.0), 1000) * u.um
        
    # High-Fidelity Blackbody Fallback
    R_sun = SOLAR_SYSTEM_DATA['Sun']['radius']
    D_earth = 1.0 * u.AU
    T_sun = SOLAR_SYSTEM_DATA['Sun']['teff']
    
    _, bb_surface = modified_blackbody(wavelengths, T_sun)
    
    # Radiance L ~ B. Flux F = pi * L (Lambertian). 
    # Irradiance E = F * (R/D)^2 = pi * B * (R/D)^2
    
    solid_angle_factor = (np.pi * (R_sun / D_earth)**2).decompose() * u.sr
    solar_flux_1au = bb_surface * solid_angle_factor
    
    if not isinstance(solar_flux_1au, u.Quantity):
        solar_flux_1au = solar_flux_1au * u.Jy 
    else:
        try:
            solar_flux_1au = solar_flux_1au.to(u.Jy, equivalencies=u.spectral_density(wavelengths))
        except Exception:
            pass # Keep original unit if conversion fails (e.g. partial dim)
        
    return wavelengths, solar_flux_1au

def calculate_composite_planet_spectrum(object_name, dist_sun, dist_obs, wavelengths=None):
    """
    Calculates hybrid spectrum (Reflected + Thermal).
    
    Reflected: F_ref = F_sun(@planet) * Albedo * Phi(alpha) * (R_pl / dist_obs)^2
    Thermal:   F_th  = B_lambda(T_eq) * pi * (R_pl / dist_obs)^2
    
    For Absolute Flux (10pc), set dist_obs=10pc, Phase=0.
    """
    if wavelengths is None:
        wavelengths = np.logspace(np.log10(0.1), np.log10(300.0), 1000) * u.um
        
    props = SOLAR_SYSTEM_DATA.get(object_name.capitalize())
    if not props:
        return wavelengths, np.zeros_like(wavelengths.value) * u.Jy
        
    R_pl = props['radius']
    T_pl = props['teff']
    Albedo = props['albedo']
    
    phi_alpha = 1.0 # Assumption: Full Phase (valid for Absolute, approx for others)
    
    # 0. Safety Check for "Observer on Surface" (Earth observing Earth)
    # If dist_obs is effectively 0, flux explodes.
    if dist_obs < 1e-5 * u.AU:
        # Return 0 to avoid artifacts in log-log plots
        return wavelengths, np.zeros_like(wavelengths.value) * u.Jy

    # 1. Thermal Component
    # F_obs = pi * B_lambda(T) * (R/D)^2
    _, bb_surf = modified_blackbody(wavelengths, T_pl)
    geom_factor_th = (np.pi * (R_pl / dist_obs)**2).decompose() * u.sr
    thermal_flux = bb_surf * geom_factor_th
    
    # 2. Reflected Component
    # F_ref_obs = F_sun_at_planet * Albedo * Phi * (R_pl/dist_obs)^2
    # F_sun_at_planet = F_sun_1au * (1AU / dist_sun)**2
    
    T_sun = SOLAR_SYSTEM_DATA['Sun']['teff']
    R_sun = SOLAR_SYSTEM_DATA['Sun']['radius']
    _, sun_surf = modified_blackbody(wavelengths, T_sun)
    
    # Avoid div by zero for Sun (dist_sun=0)
    if dist_sun < 1e-5 * u.AU:
        # Planet is inside Sun? Or target IS Sun.
        # If target IS Sun, Reflected is 0 (it emits).
        sun_flux_at_pl = 0 * sun_surf.unit * u.sr
    else:
        geom_factor_sun = (np.pi * (R_sun / dist_sun)**2).decompose() * u.sr
        sun_flux_at_pl = sun_surf * geom_factor_sun
        
    geom_factor_ref = (Albedo * phi_alpha * (R_pl / dist_obs)**2).decompose()
    reflected_flux = sun_flux_at_pl * geom_factor_ref
    
    # Total
    total_flux = thermal_flux + reflected_flux
    
    # Convert to Jy with equivalencies
    try:
        total_flux = total_flux.to(u.Jy, equivalencies=u.spectral_density(wavelengths))
    except Exception as e:
        print(f"Warning: Flux unit conversion failed for {object_name}: {e}")
    
    return wavelengths, total_flux
