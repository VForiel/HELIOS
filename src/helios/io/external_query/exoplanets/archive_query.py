
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

def query_exoplanet_archive(planet_name):
    """
    Retrieves properties for an exoplanet from the NASA Exoplanet Archive.
    Target table: 'pscomppars' (Composite Parameters).
    """
    print(f"Querying NASA Exoplanet Archive for {planet_name}...")
    
    data = {
        'identity': {'name': planet_name, 'type': 'Exoplanet', 'aliases': []},
        'host_star': {'coordinates': {'ra': None, 'dec': None, 'frame': 'icrs'}},
        'physics': {},
        'orbit': {},
        'photometry': {},
        'sed': {'wavelength': [], 'flux': []},
        'metadata': {'source': 'NASA Exoplanet Archive'}
    }
    
    try:
        table_name = "pscomppars"
        
        # Try variations of the name
        candidates = [planet_name, planet_name.replace(' ', ''), planet_name.replace('-', ' ')]
        
        tab = None
        for cand in candidates:
            try:
                where_clause = f"pl_name='{cand}'"
                tab = NasaExoplanetArchive.query_criteria(table=table_name, where=where_clause)
                if len(tab) > 0:
                    print(f"Found exoplanet with name: '{cand}'")
                    break
            except Exception as e:
                print(f"Error querying '{cand}' with criteria: {e}")
                # Fallback to query_object (sometimes more stable)
                try:
                    tab = NasaExoplanetArchive.query_object(cand, table=table_name)
                    if len(tab) > 0:
                        print(f"Found exoplanet via query_object: '{cand}'")
                        break
                except Exception as e2:
                    print(f"Error querying '{cand}' with query_object: {e2}")
                pass
        
        if tab is None or len(tab) == 0:
             print(f"Exoplanet '{planet_name}' not found in {table_name} (Tried: {candidates}).")
             return data
             
        row = tab[0]
        
        def safe_float(val):
            if np.ma.is_masked(val) or val is None: return None
            if isinstance(val, u.Quantity):
                return val.value
            try:
                return float(val)
            except:
                return None

        # --- Coordinates ---
        if 'ra' in row.colnames and 'dec' in row.colnames:
             val = safe_float(row['ra'])
             if val is not None: data['host_star']['coordinates']['ra'] = val * u.deg
             
             val = safe_float(row['dec'])
             if val is not None: data['host_star']['coordinates']['dec'] = val * u.deg
             
        # --- Host Star Props ---
        host_map = {'st_teff': 'temperature_eff', 'st_rad': 'radius', 'st_mass': 'mass', 'sy_dist': 'distance', 'st_logg': 'logg'}
        for col, key in host_map.items():
            if col in row.colnames:
                val = safe_float(row[col])
                if val is not None:
                    if key == 'distance': val = val * u.pc
                    if key == 'radius': val = val * u.R_sun
                    if key == 'mass': val = val * u.M_sun
                    if key == 'temperature_eff': val = val * u.K
                    data['host_star'][key] = val

        # --- Physics ---
        if 'pl_rade' in row.colnames:
             val = safe_float(row['pl_rade'])
             if val is not None: data['physics']['radius'] = val * u.R_earth
             
        if 'pl_bmasse' in row.colnames:
             val = safe_float(row['pl_bmasse'])
             if val is not None: data['physics']['mass'] = val * u.M_earth
             
        if 'pl_eqt' in row.colnames:
             val = safe_float(row['pl_eqt'])
             if val is not None: data['physics']['temperature_eq'] = val * u.K
             
        if 'pl_dens' in row.colnames and not np.ma.is_masked(row['pl_dens']):
             data['physics']['density'] = float(row['pl_dens']) * u.g / u.cm**3

        # --- Orbit ---
        if 'pl_orbper' in row.colnames and not np.ma.is_masked(row['pl_orbper']):
             data['orbit']['period'] = float(row['pl_orbper']) * u.day
             
        if 'pl_orbsmax' in row.colnames and not np.ma.is_masked(row['pl_orbsmax']):
             data['orbit']['semi_major_axis'] = float(row['pl_orbsmax']) * u.AU
             
        if 'pl_orbeccen' in row.colnames and not np.ma.is_masked(row['pl_orbeccen']):
             data['orbit']['eccentricity'] = float(row['pl_orbeccen']) # Dimensionless
             
        if 'pl_orbincl' in row.colnames and not np.ma.is_masked(row['pl_orbincl']):
             data['orbit']['inclination'] = float(row['pl_orbincl']) * u.deg
             
        if 'pl_tranmid' in row.colnames and not np.ma.is_masked(row['pl_tranmid']):
             data['orbit']['transit_midpoint'] = float(row['pl_tranmid']) * u.d # Spec says JD usually, but let's check unit. Usually JD.
             # Actually pl_tranmid unit is usually BJD or JD. We label it as 'd' (days) or dimensionless JD? 
             # Astropy Quantity with u.day is safer for time.
             
        if 'pl_orblper' in row.colnames and not np.ma.is_masked(row['pl_orblper']):
             data['orbit']['argument_periastron'] = float(row['pl_orblper']) * u.deg

        # --- Photometry (Host) ---
        if 'sy_vmag' in row.colnames and not np.ma.is_masked(row['sy_vmag']):
             data['photometry']['V'] = {'value': float(row['sy_vmag']), 'unit': 'mag'}
        if 'sy_kmag' in row.colnames and not np.ma.is_masked(row['sy_kmag']):
             data['photometry']['K'] = {'value': float(row['sy_kmag']), 'unit': 'mag'}

    except Exception as e:
        print(f"NASA Archive Query Error: {e}")
        
    return data
