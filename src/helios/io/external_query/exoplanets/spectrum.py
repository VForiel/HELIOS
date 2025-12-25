
import numpy as np
from astropy import units as u
from helios.sim.spectrum import modified_blackbody

def generate_exoplanet_spectrum(physics, wavelengths=None):
    """
    Generates a simple thermal spectrum for the exoplanet based on Equilibrium Temperature.
    TODO: Add reflection model once host star coupling is improved.
    """
    if wavelengths is None:
        wavelengths = np.logspace(np.log10(0.1), np.log10(30.0), 1000) * u.um
        
    T_eq = physics.get('temperature_eq')
    R_pl = physics.get('radius')
    dist = physics.get('distance')
    
    flux = None
    
    if T_eq and R_pl and dist:
        # Thermal Emission
        # F_obs = B_lambda(T) * pi * (R / D)^2
        _, bb_surf = modified_blackbody(wavelengths, T_eq)
        solid_angle = np.pi * (R_pl / dist)**2
        flux = bb_surf * solid_angle
        
        if not isinstance(flux, u.Quantity):
            flux = flux * u.Jy
            
    return wavelengths, flux
