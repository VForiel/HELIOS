
from astroquery.jplhorizons import Horizons
from astropy import units as u
from .constants import SOLAR_SYSTEM_DATA

def query_jpl_horizons(object_name, location='@399'):
    """
    Queries JPL Horizons for a specific object to get current ephemerides.
    """
    print(f"Querying JPL Horizons for {object_name}...")
    
    # Normalize name to find ID
    name_cap = object_name.capitalize()
    static_props = SOLAR_SYSTEM_DATA.get(name_cap, {})
    target_id = static_props.get('id', object_name)
    
    # Initialize new structure (v2)
    data = {
        'identity': {'jpl_id': target_id, 'name': name_cap, 'type': 'SolarSystem'},
        'ephemeris': {
            'epoch': None,
            'coordinates': {'ra': None, 'dec': None, 'delta': None, 'r': None, 'frame': 'icrs'},
            'velocity': {'d_ra_cosdec': None, 'd_dec': None}
        },
        'physics': {
            'radius': None, # To be filled from constants or query
            'albedo': None
        },
        'orbital_elements': {},
        'sed': {'wavelength': [], 'flux': []}, # Absolute Flux only
        'metadata': {}
    }

    try:
        # Quantities: 1=Astrometry (RA, DEC, Rates), 19=Helio Range, 20=Obs Range
        # Note: getting detailed osculating elements requires a separate query type (ELEMENTS vs EPHEMERIS), 
        # but Astrometry gives us position. 
        # For now, we stick to Ephemerides for position.
        obj = Horizons(id=target_id, location=location, epochs=None)
        eph = obj.ephemerides(quantities='1,19,20')
        
        if len(eph) > 0:
            row = eph[0]
            data['ephemeris']['epoch'] = str(row.get('datetime_jd', ''))
            
            # Astrometry
            if 'RA' in row.colnames and 'DEC' in row.colnames:
                data['ephemeris']['coordinates']['ra'] = float(row['RA']) * u.deg
                data['ephemeris']['coordinates']['dec'] = float(row['DEC']) * u.deg
            
            # Rates
            if 'RA_rate' in row.colnames:
                 data['ephemeris']['velocity']['d_ra_cosdec'] = float(row['RA_rate']) * u.arcsec / u.h
            if 'DEC_rate' in row.colnames:
                 data['ephemeris']['velocity']['d_dec'] = float(row['DEC_rate']) * u.arcsec / u.h

            # Observer Distance (delta)
            if 'delta' in row.colnames:
                 data['ephemeris']['coordinates']['delta'] = float(row['delta']) * u.AU
            
            # Heliocentric Distance (r)
            if 'r' in row.colnames:
                 data['ephemeris']['coordinates']['r'] = float(row['r']) * u.AU
                 
            # Fill Physics from Static Constants (fallback)
            if name_cap in SOLAR_SYSTEM_DATA:
                props = SOLAR_SYSTEM_DATA[name_cap]
                data['physics']['radius'] = props.get('radius')
                data['physics']['temperature_eff'] = props.get('teff')
                data['physics']['albedo'] = props.get('albedo')

    except Exception as e:
        print(f"JPL Horizons Error for {object_name}: {e}")
        
    return data
