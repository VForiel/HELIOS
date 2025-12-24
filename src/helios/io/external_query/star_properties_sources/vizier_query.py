
import numpy as np
import warnings
from astropy import units as u
from astroquery.vizier import Vizier
from .constants import ZERO_POINTS, WAVELENGTHS_MAP

def query_vizier(star_name, star_data):
    """
    Queries Vizier catalogs for star photometry and updates the star_data dictionary.
    """
    coords = star_data['coordinates']
    ra = coords.get('ra')
    dec = coords.get('dec')
    
    # Needs valid coords
    if not isinstance(ra, u.Quantity) or not isinstance(dec, u.Quantity):
         return star_data

    # Coordinate object for query
    # We construct a SkyCoord from the data we have. 
    # Assumes ra/dec are astropy Quantities or convertable
    from astropy.coordinates import SkyCoord
    target_coord = SkyCoord(ra, dec, frame=coords.get('frame', 'icrs'))

    try:
        v = Vizier(columns=['*', 'e_*'], row_limit=1)
        
        # Define catalogs: (ID, mapping, type, DisplayName)
        catalogs = [
            ('II/246/out', {'J': 'Jmag', 'H': 'Hmag', 'Ks': 'Kmag'}, 'mag', '2MASS'), 
            ('II/311/wise', {'W1': 'W1mag', 'W2': 'W2mag', 'W3': 'W3mag', 'W4': 'W4mag'}, 'mag', 'WISE'), 
            ('I/355/gaiadr3', {'G': 'Gmag', 'BP': 'BPmag', 'RP': 'RPmag'}, 'mag', 'Gaia DR3'), 
            ('I/259/tyc2', {'BT': 'BTmag', 'VT': 'VTmag'}, 'mag', 'Tycho-2'), 
            ('II/312/ais', {'FUV': 'FUVmag', 'NUV': 'NUVmag'}, 'mag', 'GALEX'), 
            ('II/125/main', {'12u': 'F12', '25u': 'F25', '60u': 'F60', '100u': 'F100'}, 'flux_jy', 'IRAS'), 
            ('II/298/irc', {'9u': 'S09', '18u': 'S18'}, 'flux_jy', 'AKARI')
        ]
        
        sed_temp = star_data.get('_sed_temp', {'wavelength': [], 'flux': [], 'flux_error': [], 'band': []})
        
        for entry in catalogs:
            cat_id = entry[0]
            band_map = entry[1]
            unit_type = entry[2] if len(entry) > 2 else 'mag'
            cat_name = entry[3] if len(entry) > 3 else cat_id
            
            try:
                res = v.query_region(target_coord, radius=5*u.arcsec, catalog=cat_id)
                if len(res) > 0:
                    viz_table = res[0]
                    data_found = False
                    for band, col_name in band_map.items():
                        target_col = col_name
                        if target_col not in viz_table.colnames:
                             if band == 'Ks' and 'Kmag' in viz_table.colnames: target_col = 'Kmag'
                        
                        if target_col in viz_table.colnames:
                            val = viz_table[target_col][0]
                            if not np.ma.is_masked(val):
                                flux = 0.0
                                if unit_type == 'mag':
                                    flux = ZERO_POINTS.get(band, 3631.0) * 10**(-val/2.5)
                                elif unit_type == 'flux_jy':
                                    flux = float(val)
                                
                                sed_temp['wavelength'].append(WAVELENGTHS_MAP[band])
                                sed_temp['flux'].append(flux)
                                sed_temp['band'].append(band)
                                
                                err_flux = 0.0
                                e_col = f"e_{target_col}"
                                if e_col in viz_table.colnames:
                                     e_val = viz_table[e_col][0]
                                     if not np.ma.is_masked(e_val):
                                         if unit_type == 'mag':
                                             err_flux = flux * (np.log(10)/2.5) * float(e_val)
                                         else:
                                             err_flux = float(e_val)
                                sed_temp['flux_error'].append(err_flux)
                                data_found = True
                    
                    if data_found and cat_name not in star_data['metadata']['source']:
                        star_data['metadata']['source'].append(cat_name)
                        
            except Exception:
                 pass
        
        star_data['_sed_temp'] = sed_temp
                 
    except Exception as e:
        print(f"Warning: Failed to query Vizier for {star_name}: {e}")
    
    return star_data
