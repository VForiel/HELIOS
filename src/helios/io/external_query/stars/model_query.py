
"""
Module to query theoretical spectral models (POLLUX, Pickles, etc.) via VO services or Vizier.
Used as a fallback when observational spectra are unavailable.
"""

import numpy as np
from astropy import units as u
from astroquery.vizier import Vizier

# --- CONSTANTS ---
# Pickles (1998) - Flux calibrated stellar spectral library
PICKLES_CATALOG = "J/PASP/110/863"

# POLLUX (Palacios et al. 2010) - Synthetic stellar spectra
# Accessible via VO usually, but we check if Vizier has a mirror or use simple grids.
# Note: POLLUX full database is better accessed via SSAP (Simple Spectral Access Protocol)
# But for simplicity, we might start with Pickles which is on Vizier.


from .pickles_map import PICKLES_MAP, normalize_sptype
from helios.io.external_query.stars.constants import ZERO_POINTS

def query_pickles_model(spectral_type):
    """
    Queries the Pickles library for a template spectrum matching the spectral type.
    """
    key = normalize_sptype(spectral_type)
    if not key or key not in PICKLES_MAP:
        # Fallback: Try nearest neighbor?
        # e.g. K2.5V -> K2V or K3V
        print(f"  > Pickles: Exact type '{key}' not found. (Original: {spectral_type})")
        return None
        
    filename = PICKLES_MAP[key]
    print(f"  > Pickles: Mapped '{spectral_type}' to '{filename}'")
    
    try:
        # The Pickles standard library is at J/PASP/110/863
        # The spectra are often in sub-tables or files linked.
        # Vizier access to the actual FITS/ASCII:
        # Usually J/PASP/110/863/spectra points to the data.
        
        # A reliable way using Astroquery is to query the main list for the filename
        v = Vizier(columns=['*'], row_limit=1)
        res = v.query_constraints(catalog=PICKLES_CATALOG, atfile=filename) # atfile is column name in some catalogs?
        
        # Actually in J/PASP/110/863, the table is just a list. 
        # The spectra are available via `get_data_from_table` not directly supported by generic query unless we get URLs.
        # But `astroquery.vizier` supports `get_catalogs` which executes the query.
        
        # ALTERNATIVE: Use the specific table "J/PASP/110/863/table1" 
        # But better yet, many tools use a direct URL pattern for Pickles:
        # ftp://cdsarc.u-strasbg.fr/pub/cats/J/PASP/110/863/spectra/uk...
        
        url = f"ftp://cdsarc.u-strasbg.fr/pub/cats/J/PASP/110/863/spectra/{filename}.dat"
        # Or http
        url_http = f"https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/txt?J/PASP/110/863/spectra/{filename}.dat"
        
        import requests
        from astropy.io import ascii
        
        # We try to read it directly using astropy ascii
        # The format is simple: Wavelength (Angstrom), Flux (erg/cm2/s/A) normalized? 
        # Pickles fluxes are FLAM but normalized to V=0 usually.
        
        # NOTE: Pickles spectra are normalized to zero magnitude at V band? Or Flux density? 
        # "All flux spectra in the library are normalized to unity at 5556 A"
        
        try:
             # Just use pandas or astropy to read URL
             # Try custom request to handle potential timeouts/SSL
             r = requests.get(url_http, verify=False, timeout=10)
             if r.status_code != 200:
                 # Fallback to FTP buffer?
                 pass
             
             data = ascii.read(r.text)
             # Cols: 'wav' 'flux'
             
             wl = data['col1'] * u.Angstrom
             flux = data['col2'] * (u.erg / (u.cm**2 * u.s * u.Angstrom))
             
             return {
                 'wavelength': wl.to(u.um),
                 'flux': flux, # This is RELATIVE FLUX (Normalized to unity at 5556A)
                 'source': f"Pickles ({key})"
             }
             
        except Exception as read_err:
             print(f"  > Failed to download Pickles data: {read_err}")
             return None

    except Exception as e:
        print(f"Pickles query failed: {e}")
        
    return None

def get_model_sed(star_name, star_data):
    """
    Main entry point to retrieve a model SED based on star physics.
    Scales the model to match available V-band photometry.
    """
    physics = star_data.get('physics', {})
    sp_type = physics.get('spectral_type')
    
    sed_data = None
    
    # 1. Try Pickles
    if sp_type:
         sed_data = query_pickles_model(sp_type)
    
    # 2. Scaling
    if sed_data:
        # Pickles spectra are normalized to Unity at 5556 Angstrom.
        # We need to find the V-band flux of the star to scale this.
        
        # Check Simbad Photometry in star_data
        # We look for '_sed_temp' (raw simbad) or 'photometry' if already processed.
        # Usually query_all calls this BEFORE processing _sed_temp to photometry?
        # Let's check _sed_temp.
        
        target_flux = None
        
        # Access raw temp data
        temp = star_data.get('_sed_temp', {})
        if temp and 'band' in temp and 'flux' in temp:
             bands = temp['band']
             fluxes = temp['flux']
             
             # Look for 'V'
             if 'V' in bands:
                 idx = bands.index('V')
                 target_flux = fluxes[idx] # This is in Jy usually from Simbad query
             elif 'flux(V)' in bands: # Simbad often names it thus
                 idx = bands.index('flux(V)')
                 target_flux = fluxes[idx]
        
        if target_flux is None:
             # Try V magnitude from physics/identity if we stored it? No.
             print("  > Model retrieved but no V-band photometry found for scaling. Returning normalized.")
        else:
             # Scale!
             # Unity at 5556 A (0.5556 um).
             # We assume target_flux (V band) ~ Flux at 0.5556 um.
             print(f"  > Scaling Model to V-band flux: {target_flux:.2e} Jy")
             
             # Current flux unit of sed_data is FLAM (erg/cm2/s/A)
             # target_flux is Jy (FNU)
             
             # We need to convert Unit at 0.5556 um
             # 1 Jy = 1e-23 erg/cm2/s/Hz
             # F_lambda = F_nu * c / lambda^2
             
             # Let's convert the entire SED to Jy first? 
             wl = sed_data['wavelength'] # um
             flam = sed_data['flux'] # erg/cm2/s/A
             
             # Convert FLAM to Jy
             # 1 A = 1e-4 um
             # 1 erg/cm2/s/A = 1e4 erg/cm2/s/um
             
             # F_nu(Jy) = 3.33e4 * lambda(A)^2 * F_lambda(erg/cm2/s/A)  [Approx]
             # Let's use astropy equivalencies
             
             f_jy = flam.to(u.Jy, equivalencies=u.spectral_density(wl))
             
             # Get value at 0.5556 um
             val_at_v = np.interp(0.5556, wl.to(u.um).value, f_jy.value)
             
             if val_at_v > 0:
                 scale = target_flux.value / val_at_v # Assuming target_flux is Quantity
                 f_jy_scaled = f_jy * scale
                 
                 sed_data['flux'] = f_jy_scaled
                 sed_data['source'] += " (Scaled to V)"
        
    return sed_data

