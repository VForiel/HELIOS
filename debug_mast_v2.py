
from astroquery.mast import Observations
import warnings
import numpy as np

warnings.filterwarnings("ignore")
target = "alpha lyr"

with open("mast_debug_v2.txt", "w") as f:
    f.write(f"--- Debugging MAST for {target} ---\n")
    try:
        # Get a good chunk of data
        obs = Observations.query_object(target, radius=".005 deg")
        f.write(f"Found {len(obs)} observations.\n")
        
        # Helper to safely get unique string values
        def get_unique(table, col):
            vals = []
            for row in table:
                val = row[col]
                if np.ma.is_masked(val): continue
                vals.append(str(val))
            return list(set(vals))[:20] # limit to 20
            
        if len(obs) > 0:
            f.write(f"Cols: {obs.colnames}\n")
            
            f.write(f"\nUnique provenance_name: {get_unique(obs, 'provenance_name')}\n")
            f.write(f"Unique obs_collection: {get_unique(obs, 'obs_collection')}\n")
            f.write(f"Unique project: {get_unique(obs, 'project')}\n")
            f.write(f"Unique instrument_name: {get_unique(obs, 'instrument_name')}\n")
            f.write(f"Unique target_name: {get_unique(obs, 'target_name')}\n")

            # Find row with CALSPEC in any string column
            f.write("\n--- Searching for 'CALSPEC' ---\n")
            for row in obs:
                row_str = str(row)
                if 'CALSPEC' in row_str.upper():
                    f.write(f"FOUND MATCH:\n{row}\n")
                    break # just one sample

    except Exception as e:
        f.write(f"Error: {e}\n")
