
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
    
    # Distance <-> Parallax (Assume quantities)
    if physics.get('parallax') is not None and physics.get('distance') is None:
         plx = physics['parallax']
         if plx.value > 0:
             physics['distance'] = (1000.0 * u.mas / plx).to(u.pc)
             
    if physics.get('distance') is not None and physics.get('parallax') is None:
        dist = physics['distance']
        if dist.value > 0:
            physics['parallax'] = (1000.0 * u.pc / dist).to(u.mas)
            
    # Temperature Estimation
    if physics.get('temperature_eff') is None:
        sp_type = physics.get('spectral_type')
        if sp_type:
            physics['temperature_eff'] = estimate_temperature_from_sp_type(sp_type) * u.K
        else:
            physics['temperature_eff'] = 5778.0 * u.K # Solar fallback
            
    star_data['physics'] = physics

    # --- 2. Spectrum Generation ---
    # We want a high-res SED for plotting/simulation.
    
    T_eff = physics['temperature_eff']
    if not hasattr(T_eff, 'unit'): T_eff = T_eff * u.K
    
    # Standard Grid: 0.1 um to 30 um
    wl_grid = np.logspace(np.log10(0.1), np.log10(30.0), 1000) * u.um
    c_light = 2.99792458e8 * u.m / u.s
    
    # Raw Blackbody Shape
    _, sed_bb = modified_blackbody(wavelengths=wl_grid, temperature=T_eff, beta=0.0)
    
    sed_temp = star_data.get('photometry', {})
    final_flux_jy = np.zeros_like(wl_grid.value) * u.Jy
    
    # Calculate Scaled Blackbody (Reference)
    scaled_bb_flux = None
    
    if sed_temp and 'flux' in sed_temp and len(sed_temp['flux']) > 0:
        # Extract values for interpolation
        photo_wl = sed_temp['wavelength'].to(u.um).value
        photo_flux = sed_temp['flux'].to(u.Jy).value
        
        bb_vals = sed_bb.value
        bb_wl = wl_grid.to(u.um).value
        
        # Shape: B_lambda * lambda^2 ~ B_nu (proportional)
        curve_shape = bb_vals * (bb_wl**2)
        
        interp_vals = np.interp(photo_wl, bb_wl, curve_shape)
        
        if np.max(interp_vals) > 0:
            scale_factor = 1.0
            
            anchor_band = 'V'
            bands = sed_temp.get('bands', [])
            
            # Helper to find index
            idx = -1
            if len(bands) > 0:
                 if hasattr(bands, 'tolist'): b_list = bands.tolist()
                 else: b_list = list(bands)
                 if anchor_band in b_list: idx = b_list.index(anchor_band)
            
            if idx >= 0 and idx < len(interp_vals) and interp_vals[idx] > 0:
                 scale_factor = photo_flux[idx] / interp_vals[idx]
            else:
                 scale_factor = np.mean(photo_flux) / np.mean(interp_vals)
                 
            scaled_bb_flux = (curve_shape * scale_factor) * u.Jy

    if scaled_bb_flux is None:
         # Fallback if no photometry? Just generic BB scaling?
         # Assume arbitrary scaling or normalized to 1 Jy at peak if nothing else
         scaled_bb_flux = (sed_bb.value * (wl_grid.to(u.um).value**2)) * u.Jy


    # --- HYBRID MERGE LOGIC ---
    # Check if we already have real spectral data (e.g. from CALSPEC)
    existing_sed = star_data.get('sed', {})
    
    if existing_sed.get('wavelength') is not None and len(existing_sed['wavelength']) > 10:
        # We have real data!
        real_wl = existing_sed['wavelength'].to(u.um)
        real_flux = existing_sed['flux'].to(u.Jy)
        
        # Sort just in case
        srt = np.argsort(real_wl)
        real_wl = real_wl[srt]
        real_flux = real_flux[srt]
        
        min_real = real_wl[0]
        max_real = real_wl[-1]
        
        # 1. UV Padding (< min_real)
        uv_mask = wl_grid < min_real
        uv_wl = wl_grid[uv_mask]
        uv_flux = scaled_bb_flux[uv_mask]
        
        # Scale UV part to match the first real point to avoid jump
        if len(uv_flux) > 0 and len(real_flux) > 0:
            # Factor = Real_First / Model_At_Real_First
            # Interpolate model to min_real
            model_at_boundary = np.interp(min_real.value, wl_grid.value, scaled_bb_flux.value)
            if model_at_boundary > 0:
                scale_uv = real_flux[0].value / model_at_boundary
                uv_flux = uv_flux * scale_uv
        
        # 2. IR Padding (> max_real)
        ir_mask = wl_grid > max_real
        ir_wl = wl_grid[ir_mask]
        ir_flux = scaled_bb_flux[ir_mask]
        
        # Scale IR part to match last real point
        if len(ir_flux) > 0 and len(real_flux) > 0:
             model_at_boundary = np.interp(max_real.value, wl_grid.value, scaled_bb_flux.value)
             if model_at_boundary > 0:
                 scale_ir = real_flux[-1].value / model_at_boundary
                 ir_flux = ir_flux * scale_ir
        
        # Concatenate: UV + Real + IR
        final_wl_list = []
        final_flux_list = []
        
        if len(uv_wl) > 0:
            final_wl_list.append(uv_wl)
            final_flux_list.append(uv_flux)
            
        final_wl_list.append(real_wl)
        final_flux_list.append(real_flux)
        
        if len(ir_wl) > 0:
            final_wl_list.append(ir_wl)
            final_flux_list.append(ir_flux)
            
        final_wl = np.concatenate([x.value for x in final_wl_list]) * u.um
        final_flux = np.concatenate([x.value for x in final_flux_list]) * u.Jy
        
        star_data['sed']['wavelength'] = final_wl
        star_data['sed']['flux'] = final_flux
        star_data['sed']['frequency'] = (c_light / final_wl).to(u.Hz)
        
    else:
        # No real data, use Full Model
        star_data['sed']['wavelength'] = wl_grid
        star_data['sed']['flux'] = scaled_bb_flux
        star_data['sed']['frequency'] = (c_light / wl_grid).to(u.Hz)

    # Error is zero for model
    star_data['sed']['flux_error'] = np.zeros_like(star_data['sed']['flux'].value) * u.Jy
    
    final_flux_jy = star_data['sed']['flux'] # For next steps

    
    # --- 3. Absolute Spectrum (Flux at 10pc) ---
    d_pc = physics.get('distance')
    if d_pc is not None and d_pc.value > 0:
        # Flux_10pc = Flux_obs * (d / 10pc)^2
        factor = (d_pc.to(u.pc).value / 10.0)**2
        star_data['sed']['flux_10pc'] = final_flux_jy * factor
        
        # Also calculate for photometry points
        if sed_temp and 'flux' in sed_temp:
             star_data['photometry']['flux_10pc'] = sed_temp['flux'] * factor
    else:
        star_data['sed']['flux_10pc'] = np.zeros_like(final_flux_jy.value) * u.Jy
        if sed_temp:
             star_data['photometry']['flux_10pc'] = np.zeros_like(sed_temp.get('flux', []).value) * u.Jy
             
    # --- 4. Bolometric Luminosity Estimate ---
    # L = 4 * pi * d^2 * Integral(F_nu) d_nu
    # Or L = 4 * pi * d^2 * Integral(F_lambda) d_lambda
    if d_pc is not None and d_pc.value > 0:
         # Use trapezoidal integration
         distance_m = d_pc.to(u.m).value
         
         # Integrate over frequency (Hz)
         # F_nu is in Jy -> W/m2/Hz = 1e-26
         freqs_hz = star_data['sed']['frequency'].to(u.Hz).value
         flux_jy = star_data['sed']['flux'].to(u.Jy).value
         flux_si = flux_jy * 1e-26
         
         # sort frequency ascending for integration
         sort_idx = np.argsort(freqs_hz)
         sorted_freq = freqs_hz[sort_idx]
         sorted_flux = flux_si[sort_idx]
         
         # Area under curve = Integral(F_nu d_nu) [W/m2]
         integrated_flux = np.trapz(sorted_flux, sorted_freq)
         
         lum_watts = 4 * np.pi * (distance_m**2) * integrated_flux
         l_sun = 3.828e26
         physics['luminosity_solar'] = (lum_watts / l_sun) * u.dimensionless_unscaled
         
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

def overview(data, indent=0):
    """
    Recursively prints the structure of the star data dictionary.
    Replaces arrays/lists with their shape/length.
    Replaces None or empty with 'N/A'.
    """
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{prefix}{k}: ", end="")
            if isinstance(v, (dict, list, np.ndarray, tuple)):
                if isinstance(v, u.Quantity):
                     # Handle Quantity specifically (it might be scalar or array)
                     if v.isscalar:
                         print(v)
                     else:
                         print(f"<Quantity shape={v.shape} unit={v.unit}>")
                else:
                    print("") # New line for nested structures
                    overview(v, indent + 1)
            else:
                 if v is None:
                     print("N/A")
                 else:
                     print(v)
    elif isinstance(data, (list, tuple, np.ndarray)):
        # Check if it has a shape (numpy) or len (list)
        if isinstance(data, u.Quantity):
              if data.isscalar:
                  print(data)
              else:
                  print(f"{prefix}<Quantity shape={data.shape} unit={data.unit}>")
        elif hasattr(data, 'shape'):
             # Numpy Array
             if data.dtype.kind in ('U', 'S') and data.size < 50:
                  print(f"{prefix}{data}")
             else:
                  print(f"{prefix}<Array shape={data.shape}>")
        else:
             # List or Tuple
             if len(data) > 0 and isinstance(data[0], str):
                 print(f"{prefix}{data}")
             elif len(data) == 0:
                 print(f"{prefix}[]")
             else:
                 print(f"{prefix}<List len={len(data)}>")
    elif isinstance(data, u.Quantity):
        print(f"{prefix}{data}")
    else:
        print(f"{prefix}{data}")
