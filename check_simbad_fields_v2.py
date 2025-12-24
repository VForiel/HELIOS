
from astroquery.simbad import Simbad
import requests
import urllib3
import ssl

# SSL Hack
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
old_request = requests.Session.request
def unverified_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)
requests.Session.request = unverified_request
ssl._create_default_https_context = ssl._create_unverified_context

def list_simbad_fields():
    print("Listing Simbad fields...")
    try:
        all_fields = Simbad.list_votable_fields()
        # It's an astropy Table.
        # Print column names to guess where the field name is.
        print(f"Columns: {all_fields.colnames}")
        
        # Iterate and check
        # Assuming first column is the name
        name_col = all_fields.colnames[0]
        
        error_fields = []
        for row in all_fields:
            name = row[name_col]
            if 'error' in str(name) or 'err' in str(name):
                error_fields.append(name)
                
        print(f"Found {len(error_fields)} error fields.")
        print(error_fields[:20])
        
        print("-- Specific checks --")
        search_list = [f for f in error_fields if 'plx' in f or 'ra' in f or 'pm' in f]
        print(f"Common error fields found: {search_list[:20]}")
            
    except Exception as e:
        print(f"Error parsing fields: {e}")

if __name__ == "__main__":
    list_simbad_fields()
