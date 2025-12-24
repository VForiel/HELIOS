
from astroquery.mast import Observations
import warnings
import sys

warnings.filterwarnings("ignore")
target = "alpha lyr"

with open("mast_diag.txt", "w") as f:
    f.write(f"--- Diagnosing MAST for {target} ---\n")
    
    # 1. Broad Search
    try:
        obs = Observations.query_object(target, radius=".005 deg")
        f.write(f"Total Observations: {len(obs)}\n")
        
        if len(obs) > 0:
            f.write(f"Unique Provenance: {list(set(obs['provenance_name']))}\n")
            f.write(f"Unique Collection: {list(set(obs['obs_collection']))}\n")
            f.write(f"Unique Project: {list(set(obs['project']))}\n")
            
            # Search for CALSPEC string in any column
            found_calspec = False
            for row in obs:
                for col in row.colnames:
                    if 'CALSPEC' in str(row[col]).upper():
                        f.write(f"\nFOUND CALSPEC MATCH in row:\n{row}\n")
                        found_calspec = True
                        break
                if found_calspec: break
                
    except Exception as e:
        f.write(f"Error in broad search: {e}\n")

    # 2. Specific Criteria Tests
    criteria = [
        {'provenance_name': 'CALSPEC'},
        {'project': 'CALSPEC'},
        {'obs_collection': 'CALSPEC'},
        {'target_name': 'alpha_lyr'},
        {'target_name': 'alpha lyr'}
    ]
    
    for c in criteria:
        try:
            res = Observations.query_criteria(target_name=target, **c)
            f.write(f"\nCriteria {c}: Found {len(res)} results.\n")
        except Exception as e:
             f.write(f"\nCriteria {c}: Error {e}\n")
