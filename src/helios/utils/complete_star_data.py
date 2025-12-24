
import numpy as np
from astropy import units as u
from helios.sim.spectrum import modified_blackbody

def complete_star_data(star_data):
    """
    Calculates derived fields and constructs a complete spectrum for the star data.
    
    This function modifies the input dictionary in-place by:
    1. Estimating temperature if missing.
    2. Calculating derived physics (distance/parallax, luminosity, etc.).
    3. Generates a high-resolution synthetic spectrum (blackbody) fitted to available photometry.
    4. Calculates absolute spectral luminosity (flux at 10pc).
    
    Parameters
    ----------
    star_data : dict
        The dictionary containing star properties (physics, photometry, etc.).
        
    Returns
    -------
    dict
        The updated star_data dictionary.
    """
    
    # --- 1. Physics Completion ---
    physics = star_data.get('physics', {})
    
    # Distance <-> Parallax
    if physics.get('parallax') and not physics.get('distance'):
         plx = physics['parallax']
         if plx > 0:
             physics['distance'] = 1000.0 / plx
             
    if physics.get('distance') and not physics.get('parallax'):
        dist = physics['distance']
        if dist > 0:
            physics['parallax'] = 1000.0 / dist
            
    # Temperature Estimation
    if not physics.get('temperature_eff'):
        sp_type = physics.get('spectral_type')
        if sp_type:
            physics['temperature_eff'] = estimate_temperature_from_sp_type(sp_type)
        else:
            physics['temperature_eff'] = 5778.0 # Solar fallback
            
    star_data['physics'] = physics

    # --- 2. Spectrum Generation ---
    # We want a high-res SED for plotting/simulation.
    
    T_eff = physics['temperature_eff']
    
    # Range: 0.1 um to 30 um
    wl_grid = np.logspace(np.log10(0.1), np.log10(30.0), 1000) * u.um
    star_data['sed']['wavelength'] = wl_grid.value
    c_light = 2.99792458e8
    star_data['sed']['frequency'] = c_light / (wl_grid.value * 1e-6)
    
    # Raw Blackbody Shape
    _, sed_bb = modified_blackbody(wavelengths=wl_grid, temperature=T_eff * u.K, beta=0.0)
    
    # Normalize to Photometry
    sed_temp = star_data.get('photometry', {})
    final_flux_jy = np.zeros_like(wl_grid.value)
    
    if sed_temp and 'flux' in sed_temp and len(sed_temp['flux']) > 0:
        photo_wl = np.array(sed_temp['wavelength'])
        photo_flux = np.array(sed_temp['flux'])
        
        bb_vals = sed_bb.value
        bb_wl = wl_grid.value
        
        # Curve shape in "Jy-like" space (approximate)
        # B_nu ~ B_lambda * lambda^2
        curve_shape = bb_vals * (bb_wl**2)
        
        interp_vals = np.interp(photo_wl, bb_wl, curve_shape)
        
        if np.max(interp_vals) > 0:
            # Scale Factor strategy:
            # Prefer anchoring to V-band if available, else average ratio
            scale_factor = 1.0
            
            anchor_band = 'V'
            bands = sed_temp.get('bands', [])
            
            if anchor_band in bands:
                 # bands might be a numpy array, which doesn't support .index()
                 if isinstance(bands, np.ndarray):
                     bands_list = bands.tolist()
                 else:
                     bands_list = list(bands)
                     
                 if anchor_band in bands_list:
                     idx = bands_list.index(anchor_band)
                     if idx < len(interp_vals) and interp_vals[idx] > 0:
                         scale_factor = photo_flux[idx] / interp_vals[idx]
                     else:
                         scale_factor = np.mean(photo_flux) / np.mean(interp_vals)
                 else:
                      scale_factor = np.mean(photo_flux) / np.mean(interp_vals)
            else:
                 scale_factor = np.mean(photo_flux) / np.mean(interp_vals)
                 
            final_flux_jy = curve_shape * scale_factor
            
    # Populate SED Flux
    star_data['sed']['flux'] = final_flux_jy
    star_data['sed']['flux_error'] = np.zeros_like(final_flux_jy) # Model has no error
    
    # --- 3. Absolute Spectrum (Flux at 10pc) ---
    d_pc = physics.get('distance')
    if d_pc is not None and d_pc > 0:
        factor = (d_pc / 10.0)**2
        star_data['sed']['flux_10pc'] = final_flux_jy * factor
        
        # Also calculate for photometry points
        if sed_temp and 'flux' in sed_temp:
             star_data['photometry']['flux_10pc'] = np.array(sed_temp['flux']) * factor
    else:
        star_data['sed']['flux_10pc'] = np.zeros_like(final_flux_jy)
        if sed_temp:
             star_data['photometry']['flux_10pc'] = np.zeros_like(np.array(sed_temp.get('flux', [])))
             
    # --- 4. Bolometric Luminosity Estimate (Optional/Bonus) ---
    # Integrate generated SED? 
    # L = 4 * pi * d^2 * Integral(flux)
    if d_pc is not None and d_pc > 0:
         # simple trapezoidal integration of the model spectrum
         # flux in Jy -> convert to W/m2/Hz -> integrate over Hz?
         # Or integrate F_lambda d_lambda
         # F_lambda = F_nu * c / lambda^2
         # 1 Jy = 1e-26 W/m2/Hz
         
         freqs = star_data['sed']['frequency'] # Hz
         fluxes_nu = star_data['sed']['flux'] * 1e-26 # W/m2/Hz
         
         # Integrate F_nu d_nu
         # d_nu is negative (decreasing frequency with index if wl is increasing)
         # sort by frequency for integration
         sort_idx = np.argsort(freqs)
         sorted_freq = freqs[sort_idx]
         sorted_flux = fluxes_nu[sort_idx]
         
         integrated_flux = np.trapz(sorted_flux, sorted_freq) # W/m2
         
         lum_watts = 4 * np.pi * (d_pc * 3.086e16)**2 * integrated_flux
         l_sun = 3.828e26
         physics['luminosity_solar'] = lum_watts / l_sun
         
    return star_data

def estimate_temperature_from_sp_type(sp_type):
    """Simple mapping from spectral type to effective temperature."""
    mapping = {
        'O': 30000.0,
        'B': 20000.0, 
        'A': 8500.0, 
        'F': 6500.0,
        'G': 5700.0, 
        'K': 4500.0,
        'M': 3200.0
    }
    
    if sp_type and len(sp_type) > 0:
        first_char = sp_type[0].upper()
        if first_char in mapping:
            return mapping[first_char]
            
    return 5778.0 
