
from astroquery.mast import Observations
import warnings

warnings.filterwarnings("ignore")

target = "alpha lyr"
print(f"--- Querying MAST for {target} ---")

try:
    # Query specific target
    obs = Observations.query_object(target, radius=".005 deg")
    print(f"Found {len(obs)} observations.")
    
    if len(obs) > 0:
        # Print columns of interest for the first few rows
        print("\n--- First 5 Rows Metadata ---")
        cols = ['provenance_name', 'obs_collection', 'project', 'instrument_name', 'filters', 'target_name']
        
        # Check which cols exist
        actual_cols = [c for c in cols if c in obs.colnames]
        
        for i in range(min(5, len(obs))):
            row = obs[i]
            data = {c: row[c] for c in actual_cols}
            print(f"Row {i}: {data}")

        # Summary of unique values for whole table
        print("\n--- Unique Values ---")
        for c in actual_cols:
            vals = set(obs[c])
            # Limit print size
            if len(vals) > 20:
                print(f"{c}: {len(vals)} unique values (too many to list)")
            else:
                print(f"{c}: {vals}")

except Exception as e:
    print(f"Error: {e}")
