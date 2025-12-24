
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
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

def test_spectral_query(target):
    print(f"--- Querying Spectra for {target} ---")
    
    # ELODIE (III/218), MILES (III/252)
    # Pickles (J/PASP/110/861)
    catalogs = ['III/218/elodie', 'III/252/miles_s', 'J/PASP/110/861/flux']
    
    v = Vizier(columns=['*'])
    try:
        c = SkyCoord.from_name(target)
    except Exception as e:
        print(f"Could not resolve {target}: {e}")
        return

    for cat in catalogs:
        print(f"Querying {cat}...")
        try:
            res = v.query_region(c, radius=10*u.arcsec, catalog=cat)
            if len(res) > 0:
                print(f"Found {len(res)} tables in {cat}")
                # Inspect columns to see where flux/wavelength is
                t = res[0]
                print(f"Columns: {t.colnames}")
                # Usually these catalogs provide a link to a file or columns for Spectrum
            else:
                print(f"No match in {cat}.")
        except Exception as e:
            print(f"Error querying {cat}: {e}")

if __name__ == "__main__":
    test_spectral_query("Vega")
    test_spectral_query("Betelgeuse")
