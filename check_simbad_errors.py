
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

def check_errors():
    print("Checking Simbad errors field...")
    Simbad.add_votable_fields('errors')
    # Also add flux errors specifically if needed?
    Simbad.add_votable_fields('flux_error(V)') 
    
    res = Simbad.query_object("Vega")
    if res:
        print("Columns:", res.colnames)
        print("First row:", res[0])
    
if __name__ == "__main__":
    check_errors()
