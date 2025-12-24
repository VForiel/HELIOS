
from helios.io.external_query.stars.query_all import get_star_properties
import numpy as np

# Test with Sirius (alpha CMa) - A well known CALSPEC standard
# Test with 109 Vir - A standard often used but less famous than Vega
targets = ["Sirius", "109 Vir"]

for name in targets:
    print(f"\n--- Testing {name} ---")
    star_data = {
        'identity': {'name': name},
        'metadata': {'source': []},
        'sed': {'wavelength': [], 'flux': []}
    }
    
    try:
        data = query_stsci_calspec(name, star_data)
        
        pts = len(data['sed']['wavelength'])
        if pts > 100:
            print(f"SUCCESS: Retrieved {pts} points for {name}.")
            print(f"Sources: {data['metadata']['source']}")
        else:
            print(f"FAILED: No spectrum found for {name}.")
            
    except Exception as e:
        print(f"ERROR for {name}: {e}")
