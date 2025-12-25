
import os
import time
import json
from datetime import datetime
from astropy import units as u
import numpy as np

from .jpl_query import query_jpl_horizons
from .spectrum import get_solar_spectrum, get_real_planet_spectrum
from .serialization import serialize_solar_data, deserialize_solar_data

def get_solar_system_properties(object_name, complete_data=False, force=False):
    """
    Main entry point for Solar System objects.
    Orchestrates Cache -> Query -> Spectrum -> Save.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    safe_name = object_name.capitalize()
    cache_file = os.path.join(cache_dir, f"{safe_name}.json")
    
    data = None
    cache_valid = False
    
    # 1. Load Cache (TTL 1 week)
    if not force and os.path.exists(cache_file):
        max_age = 7 * 24 * 3600
        file_age = time.time() - os.path.getmtime(cache_file)
        
        if file_age < max_age:
            try:
                print(f"Loading '{safe_name}' from SS cache...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    data = deserialize_solar_data(raw)
                cache_valid = True
            except Exception as e:
                print(f"SS cache load failed: {e}")
                
    if not cache_valid:
        print(f"Querying new data for {safe_name}...")
        
        # 2. Query JPL
        data = query_jpl_horizons(safe_name)
        data['identity']['name'] = safe_name
        data['metadata'] = {'sources': ['JPL Horizons'], 'retrieved_at': datetime.now().isoformat()}
         
        # 3. Generate Spectrum
        if safe_name == 'Sun':
             wl, flux = get_solar_spectrum()
             data['sed']['wavelength'] = wl
             data['sed']['flux'] = flux
             
             # Calculate Flux @ 10pc for Absolute usage
             factor = ((1.0 * u.AU / (10.0 * u.pc))**2).decompose()
             data['sed']['flux'] = flux * factor # Overwrite 1AU flux with 10pc Flux
             data['sed']['source'] = "ASTM E-490 (Scaled to 10pc)"
             
        else:
             # Planets
             # Get Distances (New Structure)
             coords = data['ephemeris']['coordinates']
             dist_obs = coords.get('delta')
             if dist_obs is None: dist_obs = 1.0 * u.AU
             
             dist_sun = coords.get('r')
             if dist_sun is None: dist_sun = 1.0 * u.AU
             
             # Calculate Absolute Flux (Reference at 10pc)
             # Note: We NO LONGER store 'observed' flux in the cache, only Absolute.
             # The user asked for "Absolute SED" only.
             
             dist_10pc = 10.0 * u.pc
             # Try to get REAL spectrum (No Synthetic)
             # If unavailable, returns None, None - and we cache EMPTY to avoid re-querying or just leave it empty.
             wl_out, flux_abs = get_real_planet_spectrum(safe_name, dist_sun, dist_10pc)
             
             if wl_out is not None and len(wl_out) > 0:
                 data['sed']['wavelength'] = wl_out
                 data['sed']['flux'] = flux_abs
                 data['sed']['source'] = f"Real Observed Data (Scaled to 10pc)"
             else:
                 # NO SYNTHETIC DATA FALLBACK
                 print(f"No real spectrum found for {safe_name}. Cache will contain empty SED.")
                 data['sed'] = {'wavelength': [], 'flux': []}
        
        # 4. Save Cache
        try:
             serial = serialize_solar_data(data)
             with open(cache_file, 'w', encoding='utf-8') as f:
                 json.dump(serial, f, separators=(',', ':'))
        except Exception as e:
             print(f"Failed to save SS cache: {e}")
             
    return data
