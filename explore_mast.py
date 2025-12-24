
from astroquery.mast import Observations
import warnings

target = "Vega"
print(f"--- Querying MAST for {target} ---")

try:
    # Query blindly by object name
    obs = Observations.query_object(target, radius=".005 deg") # 18 arcsec
    print(f"Found {len(obs)} observations.")
    
    if len(obs) > 0:
        # Check available provenances
        provs = set(obs['provenance_name'])
        print(f"Available Provenances: {provs}")
        
        # Check available instruments
        instrs = set(obs['instrument_name'])
        print(f"Available Instruments: {instrs}")
        
        # Check available collections
        colls = set(obs['obs_collection'])
        print(f"Available Collections: {colls}")
        
        # Try to find one that looks like a standard
        for p in provs:
            if 'CAL' in str(p).upper():
                print(f"Potential CAL provenance: {p}")
                
except Exception as e:
    print(f"Error: {e}")
