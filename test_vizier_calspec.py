
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

target = SkyCoord.from_name("Vega")
print(f"Target: {target}")

# query J/AJ/157/229 (CALSPEC Bohlin 2019)
# Try different radii
for r in [5, 30, 60]:
    print(f"\n--- Radius: {r} arcsec ---")
    try:
        # Columns: Need lambda (Wavelength) and Flux
        v = Vizier(columns=['*'], row_limit=50) 
        res = v.query_region(target, radius=r*u.arcsec, catalog="III/202")
        print(f"Found {len(res)} tables.")
        for t in res:
            print(f"Table {t.meta.get('name')}: {len(t)} rows")
            print(f"Cols: {t.colnames}")
            if len(t) > 0:
                print(f"Row 0: {t[0]}")
    except Exception as e:
        print(f"Error: {e}")
