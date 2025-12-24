
import numpy as np
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from .constants import ZERO_POINTS, WAVELENGTHS_MAP

def query_simbad(star_name, star_data):
    """
    Queries Simbad for star properties and updates the star_data dictionary.
    """
    try:
        # Add fields to Simbad query
        # Requesting a wide range of fluxes available in Simbad
        flux_fields = [
            'flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', # Johnson/Cousins
            'flux(J)', 'flux(H)', 'flux(K)', # 2MASS (Direct from Simbad too)
            'flux(u)', 'flux(g)', 'flux(r)', 'flux(i)', 'flux(z)' # SDSS
        ]
        
        # We start fresh with Simbad fields to ensure we don't accumulate from previous calls if Simbad is stateful globally
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields(
            'ra(d)', 'dec(d)', 
            'pmra', 'pmdec', 
            'plx', 'sptype', 
            *flux_fields
        )
        
        simbad_res = Simbad.query_object(star_name)
        
        if simbad_res is None:
             print(f"Warning: Star '{star_name}' not found in Simbad.")
             return star_data
        
        table = simbad_res[0] 
        
        # Populate Identity
        star_data['identity']['simbad_id'] = str(table['MAIN_ID']) if 'MAIN_ID' in table.colnames else star_name

        # Populate Coordinates
        if 'RA_d' in table.colnames and 'DEC_d' in table.colnames:
             star_data['coordinates']['ra'] = float(table['RA_d']) * u.deg
             star_data['coordinates']['dec'] = float(table['DEC_d']) * u.deg
        
        # Re-resolve coords
        try:
            coords = SkyCoord.from_name(star_name)
            # Prefer SkyCoord if Simbad table is missing precise degrees or purely for robustness
            star_data['coordinates']['ra'] = coords.ra
            star_data['coordinates']['dec'] = coords.dec
        except Exception:
            pass # Use Simbad RA/DEC if from_name fails (though from_name uses simbad/sesame usually)

        # Populate Kinematics
        if 'pmra' in table.colnames and not np.ma.is_masked(table['pmra']):
            star_data['kinematics']['pm_ra'] = float(table['pmra']) * u.mas/u.yr
        if 'pmdec' in table.colnames and not np.ma.is_masked(table['pmdec']):
            star_data['kinematics']['pm_dec'] = float(table['pmdec']) * u.mas/u.yr
        if 'rv_value' in table.colnames and not np.ma.is_masked(table['rv_value']):
             # rv_value is usually in km/s in Simbad default output for 'rv_value' field? 
             # Simbad 'rv_value' is Radial velocity. Default unit often km/s.
            star_data['kinematics']['radial_velocity'] = float(table['rv_value']) * u.km/u.s

        # Populate Physics
        if 'plx_value' in table.colnames and not np.ma.is_masked(table['plx_value']):
            plx = float(table['plx_value'])
            star_data['physics']['parallax'] = plx * u.mas
            if plx > 0:
                star_data['physics']['distance'] = (1000.0 / plx) * u.pc
        
        if 'sp_type' in table.colnames and not np.ma.is_masked(table['sp_type']):
            star_data['physics']['spectral_type'] = str(table['sp_type'])
            
        # Update metadata
        if 'Simbad' not in star_data['metadata']['sources']:
             star_data['metadata']['sources'].append('Simbad')

        # --- Simbad Harvest Photometry ---
        simbad_bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'u', 'g', 'r', 'i', 'z']
        
        sed_temp = star_data.get('_sed_temp', {'wavelength': [], 'flux': [], 'flux_error': [], 'band': []})
        
        for band in simbad_bands:
            cols_to_check = [f"FLUX_{band}", band, f"flux({band})"]
            col = None
            for c in cols_to_check:
                if c in table.colnames:
                    col = c
                    break
            
            if col:
                val = table[col]
                if not np.ma.is_masked(val):
                     if band in ZERO_POINTS:
                         flux = ZERO_POINTS[band] * 10**(-val/2.5)
                         sed_temp['wavelength'].append(WAVELENGTHS_MAP[band])
                         sed_temp['flux'].append(flux)
                         sed_temp['band'].append(band)
                         
                         err_flux = 0.0
                         err_col_options = [f"FLUX_ERROR_{band}", f"flux_error({band})"]
                         for ec in err_col_options:
                             if ec in table.colnames:
                                  e_val = table[ec]
                                  if not np.ma.is_masked(e_val):
                                      err_flux = flux * (np.log(10)/2.5) * float(e_val)
                                      break
                         sed_temp['flux_error'].append(err_flux)
        
        star_data['_sed_temp'] = sed_temp

    except Exception as e:
        print(f"Error querying Simbad for {star_name}: {e}")
        
    return star_data
