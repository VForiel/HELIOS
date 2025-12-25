
import os
import time
import json
from datetime import datetime
from .archive_query import query_exoplanet_archive
from .spectrum import generate_exoplanet_spectrum
from .serialization import serialize_exo_data, deserialize_exo_data
from astropy import units as u

def get_exoplanet_properties(planet_name, complete_data=False, force=False):
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Sanitize name
    safe_name = "".join([c for c in planet_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    cache_file = os.path.join(cache_dir, f"{safe_name}.json")
    
    data = None
    cache_valid = False
    
    if not force and os.path.exists(cache_file):
        max_age = 30 * 24 * 3600 # 30 days
        if (time.time() - os.path.getmtime(cache_file)) < max_age:
            try:
                print(f"Loading '{planet_name}' from Exoplanet cache...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = deserialize_exo_data(json.load(f))
                cache_valid = True
            except Exception as e:
                print(f"Exo cache load failed: {e}")
                
    if not cache_valid:
        data = query_exoplanet_archive(planet_name)
        if not data.get('physics') and data['host_star'].get('coordinates', {}).get('ra') is None:
            # Not found
            print(f"Exoplanet '{planet_name}' not found.")
            return None
            
        # NOTE: Synthetic spectrum generation REMOVED as per user requirement.
        # Cache contains only observed parameters.
        # data['sed'] remains empty unless populated by future observational queries.
        
        # Save Cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serialize_exo_data(data), f, separators=(',', ':'))
        except Exception as e:
            print(f"Exo cache save failed: {e}")
            
    return data
