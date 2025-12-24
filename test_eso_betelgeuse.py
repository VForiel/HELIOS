
from helios.io.external_query.stars.query_all import get_star_properties
import os

def test():
    print("Testing ESO Query for Betelgeuse...")
    
    # Force new query to ignore existing cache/synthetic data
    data = get_star_properties("Betelgeuse", complete_data=True, plot=False, force=True)
    
    if data:
        sources = data['metadata']['sources']
        print("Sources:", sources)
        
        has_eso = any("ESO" in s for s in sources)
        if has_eso:
            print("SUCCESS: ESO source found!")
            sed = data['sed']
            print(f"Spectrum Points: {len(sed['wavelength'])}")
            
            # Verify range
            if len(sed['wavelength']) > 0:
                w_min = sed['wavelength'].min()
                w_max = sed['wavelength'].max()
                print(f"Range: {w_min} - {w_max}")
        else:
            print("FAILURE: ESO source NOT found.")
    else:
        print("FAILURE: No data returned.")

if __name__ == '__main__':
    test()
