
from astroquery.simbad import Simbad

def list_simbad_fields():
    print("Listing Simbad fields...")
    all_fields = Simbad.list_votable_fields()
    
    # Filter for error-related fields
    error_fields = [f for f in all_fields if 'error' in f or 'err' in f]
    print(f"Found {len(error_fields)} error fields.")
    for f in error_fields[:20]: # Print first 20
        print(f)
        
    print("-- Specific checks --")
    print(f"plx_error: {'plx_error' in all_fields}")
    print(f"error(plx): {'error(plx)' in all_fields}")


if __name__ == "__main__":
    list_simbad_fields()
