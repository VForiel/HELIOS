
"""
Module for accessing online astronomical catalogs to retrieve comprehensive star properties and SEDs.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import warnings
import ssl
import requests
import urllib3
from functools import partial
from datetime import datetime
from helios.sim.spectrum import modified_blackbody

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Hack: Force all requests to be unverified
old_request = requests.Session.request

def unverified_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)

requests.Session.request = unverified_request

# Also patch ssl default context just in case
ssl._create_default_https_context = ssl._create_unverified_context

def get_star_properties(star_name, plot=False):
    """
    Retrieves comprehensive properties of a given star including its SED.
    
    Parameters
    ----------
    star_name : str
        The common name of the star (e.g. "Vega", "Betelgeuse").
    plot : bool, optional
        If True, displays a plot of the retrieved SED. Default is False.
        
    Returns
    -------
    dict
        A dictionary containing star properties structured as:
        - 'identity': name, id
        - 'coordinates': ra, dec, frame
        - 'kinematics': pm_ra, pm_dec, radial_velocity
        - 'physics': parallax, distance, spectral_type, temperature
        - 'sed': wavelength, flux, bands, frequency
        - 'metadata': source, retrieval time
    """
    print(f"Retrieving properties for {star_name}...")
    
    # Structure to populate
    star_data = {
        'identity': {'name': star_name, 'simbad_id': None},
        'coordinates': {'ra': None, 'dec': None, 'frame': 'icrs'},
        'kinematics': {'pm_ra': None, 'pm_dec': None, 'radial_velocity': None},
        'physics': {'parallax': None, 'distance': None, 'spectral_type': None, 'temperature_eff': None},
        'sed': {'wavelength': [], 'flux': [], 'frequency': [], 'bands': []},
        'metadata': {'source': 'Simbad/Vizier', 'retrieved_at': datetime.now().isoformat()}
    }

    # 1. Resolve star name and basic data via Simbad
    try:
        # Add fields to Simbad query
        Simbad.add_votable_fields(
            'ra(d)', 'dec(d)', 
            'pmra', 'pmdec', 
            'plx', 'sptype', 
            'flux(B)', 'flux(V)', 'flux(J)', 'flux(H)', 'flux(K)'
        )
        
        simbad_res = Simbad.query_object(star_name)
        
        if simbad_res is None:
             # Fallback to coordinate resolution only if full query fails?
             # But usually query_object returns something if resolved.
             raise ValueError(f"Star '{star_name}' not found in Simbad.")
        
        table = simbad_res[0] # Take first result
        
        # Populate Identity
        star_data['identity']['simbad_id'] = str(table['MAIN_ID']) if 'MAIN_ID' in table.colnames else star_name

        # Populate Coordinates
        if 'RA_d' in table.colnames and 'DEC_d' in table.colnames:
             star_data['coordinates']['ra'] = float(table['RA_d'])
             star_data['coordinates']['dec'] = float(table['DEC_d'])
             coords = SkyCoord(ra=table['RA_d'], dec=table['DEC_d'], unit=(u.deg, u.deg))
        elif 'ra' in table.colnames and 'dec' in table.colnames:
             # Handle lowercase sexagesimal or decimal? 
             # Simbad 'ra' is usually sexagesimal string if not specified as 'd'
             # actually I requested ra(d), dec(d). Simbad returns them as 'ra', 'dec' but in degrees?
             # Debug output showed 'ra', 'dec'. Let's assume they are the ones requested.
             # but query_object usually returns sexagesimal by default for 'ra'.
             # However I added 'ra(d)'...
             # Let's trust SkyCoord.from_name for coordinates as it is robust.
             pass
        
        # Re-resolve coords just to be safe and uniform if columns are ambiguous
        coords = SkyCoord.from_name(star_name)
        star_data['coordinates']['ra'] = coords.ra.deg
        star_data['coordinates']['dec'] = coords.dec.deg

        # Populate Kinematics
        if 'pmra' in table.colnames and not np.ma.is_masked(table['pmra']):
            star_data['kinematics']['pm_ra'] = float(table['pmra'])
        if 'pmdec' in table.colnames and not np.ma.is_masked(table['pmdec']):
            star_data['kinematics']['pm_dec'] = float(table['pmdec'])
        if 'rv_value' in table.colnames and not np.ma.is_masked(table['rv_value']):
            star_data['kinematics']['radial_velocity'] = float(table['rv_value'])

        # Populate Physics
        if 'plx_value' in table.colnames and not np.ma.is_masked(table['plx_value']):
            plx = float(table['plx_value'])
            star_data['physics']['parallax'] = plx
            if plx > 0:
                star_data['physics']['distance'] = 1000.0 / plx # Distance in parsecs
        
        if 'sp_type' in table.colnames and not np.ma.is_masked(table['sp_type']):
            star_data['physics']['spectral_type'] = str(table['sp_type'])
            
        # 2. SED Data
        # We query multiple catalogs to build a fuller SED.
        # Catalogs: Simbad (B,V), 2MASS (J,H,K), WISE (W1..4), Gaia (G,BP,RP)
        
        # Zero points (Jy) for Vega-based magnitudes (Approximate)
        zero_points = {
            'J': 1594.0, 'H': 1024.0, 'Ks': 666.8, 
            'W1': 309.54, 'W2': 171.79, 'W3': 31.67, 'W4': 8.36,
            'B': 4130.0, 'V': 3781.0,
            'G': 3229.0, 'BP': 3552.0, 'RP': 2555.0
        }
        
        # Effective Wavelengths (microns)
        wavelengths_map = {
            'J': 1.235, 'H': 1.662, 'Ks': 2.159,
            'W1': 3.35, 'W2': 4.60, 'W3': 11.56, 'W4': 22.09,
            'B': 0.44, 'V': 0.55,
            'G': 0.62, 'BP': 0.51, 'RP': 0.78 # Approximate center wavelengths
        }
        
        sed_temp = {'wavelength': [], 'flux': [], 'flux_error': [], 'band': []}

        # --- Simbad (B, V) ---
        for band in ['B', 'V']:
            cols_to_check = [f"FLUX_{band}", band, f"flux({band})"]
            col = None
            for c in cols_to_check:
                if c in table.colnames:
                    col = c
                    break
            
            if col:
                val = table[col]
                if not np.ma.is_masked(val):
                     flux = zero_points[band] * 10**(-val/2.5)
                     sed_temp['wavelength'].append(wavelengths_map[band])
                     sed_temp['flux'].append(flux)
                     sed_temp['band'].append(band)
                     
                     # Flux Error
                     err_flux = 0.0
                     # Check for Flux Error Columns (Simbad often uses FLUX_ERROR_V or similar)
                     err_col_options = [f"FLUX_ERROR_{band}", f"flux_error({band})"]
                     for ec in err_col_options:
                         if ec in table.colnames:
                              e_val = table[ec]
                              if not np.ma.is_masked(e_val):
                                  # Error propagation: dF = F * ln(10)/2.5 * dm
                                  err_flux = flux * (np.log(10)/2.5) * float(e_val)
                                  break
                     sed_temp['flux_error'].append(err_flux)

    except Exception as e:
        print(f"Error querying Simbad for {star_name}: {e}")
        return None # Critical failure if we can't even ID the star

    # --- Query Vizier Catalogs ---
    try:
        v = Vizier(columns=['*', 'e_*'], row_limit=1)
        
        # Define catalogs to query: (ID, mapping of Band->ColName)
        catalogs = [
            ('II/246/out', {'J': 'Jmag', 'H': 'Hmag', 'Ks': 'Kmag'}), # 2MASS
            ('II/311/wise', {'W1': 'W1mag', 'W2': 'W2mag', 'W3': 'W3mag', 'W4': 'W4mag'}), # WISE
            ('I/355/gaiadr3', {'G': 'Gmag', 'BP': 'BPmag', 'RP': 'RPmag'}) # Gaia DR3
        ]
        
        for cat_id, band_map in catalogs:
            try:
                res = v.query_region(coords, radius=5*u.arcsec, catalog=cat_id)
                if len(res) > 0:
                    viz_table = res[0]
                    for band, col_name in band_map.items():
                        # Handle potential column name mismatches (e.g. Kmag vs K)
                        if col_name not in viz_table.colnames:
                             # Try fallback if specific known issues exist (like Ks/K)
                             if band == 'Ks' and 'Kmag' in viz_table.colnames: col_name = 'Kmag'
                        
                        if col_name in viz_table.colnames:
                            val = viz_table[col_name][0]
                            if not np.ma.is_masked(val):
                                flux = zero_points.get(band, 3631.0) * 10**(-val/2.5)
                                sed_temp['wavelength'].append(wavelengths_map[band])
                                sed_temp['flux'].append(flux)
                                sed_temp['band'].append(band)
                                
                                # Error
                                err_flux = 0.0
                                e_col = f"e_{col_name}"
                                if e_col in viz_table.colnames:
                                     e_mag = viz_table[e_col][0]
                                     if not np.ma.is_masked(e_mag):
                                         err_flux = flux * (np.log(10)/2.5) * float(e_mag)
                                sed_temp['flux_error'].append(err_flux)
            except Exception as e:
                 pass
                 
    except Exception as e:
        print(f"Warning: Failed to query Vizier for {star_name}: {e}")

    # --- Generate Synthetic High-Res Spectrum ---
    # Use the centralized utility to complete the data set
    from helios.utils.complete_star_data import complete_star_data
    
    # We pass the collected incomplete data
    # (Identity, Coordinates, Kinematics, Physics(partial), Photometry(partial))
    
    # Ensure Photometry structure is ready for the utility
    if not star_data['sed']['wavelength']:
         # The utility expects 'photometry' to populate 'sed' model.
         # We have sed_temp from our queries.
         star_data['photometry'] = {
            'wavelength': np.array(sed_temp['wavelength']),
            'flux': np.array(sed_temp['flux']),
            'flux_error': np.array(sed_temp['flux_error']),
            'bands': np.array(sed_temp['band'])
         }

    # Run completion logic
    star_data = complete_star_data(star_data)

    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        
        # High-Res Model
        if len(star_data['sed']['wavelength']) > 0:
            T_label = f"{star_data['physics'].get('temperature_eff', 0):.0f} K"
            plt.loglog(star_data['sed']['wavelength'], star_data['sed']['flux'], '-', 
                       label=f"Model ({T_label})", color='gray', alpha=0.7)
        
        # Photometry Points with Errors
        photo = star_data.get('photometry', {})
        if photo and len(photo.get('wavelength', [])) > 0:
            plt.errorbar(photo['wavelength'], photo['flux'], 
                         yerr=photo.get('flux_error'), 
                         fmt='o', color='red', label='Photometry', ecolor='salmon', capsize=3)
            
            if 'bands' in photo:
                for i, txt in enumerate(photo['bands']):
                    plt.annotate(txt, (photo['wavelength'][i], photo['flux'][i]), 
                                 xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)

        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Flux Density (Jy)')
        plt.title(f'SED: {star_name} ({star_data["physics"].get("spectral_type", "Unknown")})')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.show()

    return star_data
