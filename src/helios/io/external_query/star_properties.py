
"""
Module for accessing online astronomical catalogs to retrieve comprehensive star properties and SEDs.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from datetime import datetime
import urllib3
import ssl
import requests

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Hack: Force all requests to be unverified
old_request = requests.Session.request
def unverified_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)
requests.Session.request = unverified_request
ssl._create_default_https_context = ssl._create_unverified_context

from helios.io.external_query.star_properties_sources import query_simbad, query_vizier, query_stsci_calspec, query_vizier_spectra

def get_star_properties(star_name, complete_data=False, plot=False):
    """
    Retrieves comprehensive properties of a given star including its SED.
    """
    print(f"Retrieving properties for {star_name}...")
    
    # Structure to populate
    star_data = {
        'identity': {'name': star_name, 'simbad_id': None},
        'coordinates': {'ra': None, 'dec': None, 'frame': 'icrs'},
        'kinematics': {'pm_ra': None, 'pm_dec': None, 'radial_velocity': None},
        'physics': {'parallax': None, 'distance': None, 'spectral_type': None, 'temperature_eff': None},
        'sed': {'wavelength': [], 'flux': [], 'frequency': [], 'bands': []},
        'metadata': {'source': [], 'retrieved_at': datetime.now().isoformat()},
        # Temporary storage for raw photometry before formatting
        '_sed_temp': {'wavelength': [], 'flux': [], 'flux_error': [], 'band': []}
    }

    # 1. Query Simbad
    star_data = query_simbad(star_name, star_data)
    
    # 2. Query Vizier (Photometry)
    if star_data['coordinates']['ra'] is not None:
        star_data = query_vizier(star_name, star_data)
        
    # 3. Query STScI (MAST/CALSPEC) for standard spectrum
    # Check if empty (list or array)
    if len(star_data['sed']['wavelength']) == 0:
         star_data = query_stsci_calspec(star_name, star_data)

    # 4. Fallback to Vizier Spectra (e.g. Burnashev)
    # Check if still empty
    if len(star_data['sed']['wavelength']) == 0:
         star_data = query_vizier_spectra(star_name, star_data)

    # Process retrieved photometry into Star Data Structure
    sed_temp = star_data.pop('_sed_temp')
    
    if len(sed_temp['wavelength']) > 0:
         star_data['photometry'] = {
            'wavelength': np.array(sed_temp['wavelength']) * u.um,
            'flux': np.array(sed_temp['flux']) * u.Jy,
            'flux_error': np.array(sed_temp['flux_error']) * u.Jy,
            'bands': np.array(sed_temp['band'])
         }

    # --- Generate Synthetic High-Res Spectrum ---
    if complete_data:
        # Use the centralized utility to complete the data set
        from helios.utils.data_completion.star import complete_star_data
        
        # Ensure Photometry structure is ready for the utility if not already (it is above)
        if len(star_data['sed']['wavelength']) == 0 and 'photometry' in star_data:
             pass # Photometry is set

        # Run completion logic
        star_data = complete_star_data(star_data)

    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        
        # High-Res Model
        if len(star_data['sed']['wavelength']) > 0:
            val = star_data['physics'].get('temperature_eff')
            T_label = f"{val.value:.0f} K" if hasattr(val, 'value') else f"{val} K"
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
                     # Handle quantities for plotting
                    wl_val = photo['wavelength'][i].value if hasattr(photo['wavelength'][i], 'value') else photo['wavelength'][i]
                    flux_val = photo['flux'][i].value if hasattr(photo['flux'][i], 'value') else photo['flux'][i]
                    plt.annotate(txt, (wl_val, flux_val), 
                                 xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)

        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Flux Density (Jy)')
        # Handle spectral type safely
        sp = star_data["physics"].get("spectral_type", "Unknown")
        plt.title(f'SED: {star_name} ({sp})')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.show()

    return star_data
