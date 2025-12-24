
from helios.io.external_query.stars.query_all import get_star_properties
import numpy as np

def test():
    print("Testing Multi-Source Merging for Betelgeuse...")
    
    # Force new query to trigger multi-source logic
    data = get_star_properties("Betelgeuse", complete_data=True, plot=False, force=True)
    
    if data:
        sources = data['metadata']['sources']
        print(f"Sources found: {sources}")
        
        # Check if we have multiple spectral sources
        spectral_sources = [s for s in sources if 'Simbad' not in s and '2MASS' not in s and 'WISE' not in s and 'Gaia' not in s]
        print(f"Spectral Sources: {spectral_sources}")
        
        sed = data['sed']
        print(f"Total Merged Spectrum Points: {len(sed['wavelength'])}")
        
        if len(sed['wavelength']) > 0:
            print(f"Wavelength Range: {sed['wavelength'].min()} - {sed['wavelength'].max()}")
            
        if len(spectral_sources) > 1:
             print("SUCCESS: Multiple spectral sources were merged!")
        elif len(spectral_sources) == 1:
             print("PARTIAL SUCCESS: Only one spectral source found (maybe normal if others are empty).")
        else:
             print("FAILURE: No spectral sources found.")
             
    else:
        print("FAILURE: No data returned.")

if __name__ == '__main__':
    test()
