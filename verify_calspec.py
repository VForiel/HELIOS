
from helios.io.external_query.star_properties_sources.stsci_query import query_stsci_calspec
from datetime import datetime
import numpy as np

# Mock star_data structure
star_data = {
    'identity': {'name': 'Vega'},
    'metadata': {'source': []},
    'sed': {'wavelength': [], 'flux': []}
}

print("Running query_stsci_calspec for Vega...")
try:
    data = query_stsci_calspec("Vega", star_data)
    
    if len(data['sed']['wavelength']) > 100:
        print(f"VERIFICATION SUCCESS: Retrieved {len(data['sed']['wavelength'])} points.")
        print(f"Sources: {data['metadata']['source']}")
    else:
        print("VERIFICATION FAILED: No data retrieved.")
except Exception as e:
    print(f"VERIFICATION ERROR: {e}")
