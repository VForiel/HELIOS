
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

target = SkyCoord.from_name("Vega")

# Candidate catalogs
# J/AJ/157/229: CALSPEC Bohlin 2019
# III/202: Burnashev Spectrophotometry
catalogs = ["J/AJ/157/229", "III/202"]

print(f"--- Inspecting Catalogs for {target} ---")

for cat_id in catalogs:
    print(f"\n--- Catalog: {cat_id} ---")
    try:
        v = Vizier(columns=['**'], row_limit=5) # Request all columns
        res = v.query_region(target, radius=5*u.arcsec, catalog=cat_id)
        
        if len(res) > 0:
            for table in res:
                print(f"Table: {table.meta.get('name')}")
                print(f"Desc: {table.description}")
                print(f"Columns: {table.colnames[:10]}...") 
                print(f"Data (1st row): {table[0]}")
        else:
            print("  -> No data found for Vega in this catalog.")
            
    except Exception as e:
        print(f"  -> Error: {e}")
