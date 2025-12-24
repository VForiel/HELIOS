
from astroquery.eso import Eso
from astroquery.utils.tap.core import TapPlus

def search_eso_astroquery():
    print("Initializing ESO query...")
    eso = Eso()
    eso.ROW_LIMIT = 50
    # Login might be needed for proprietary, but Betelgeuse data should be public.
    # We look for "Science" products (Phase 3) ideally.
    
    target = "alpha Ori"
    print(f"Querying ESO Main (Instrument/Raw/Processed?) for {target}...")
    
    try:
        # Generic query to see what's there
        table = eso.query_main(target=target, coord_sys="J2000")
        if table:
            print(f"Found {len(table)} records.")
            print("Columns:", table.colnames)
            # Filter for likely spectral instruments
            # XSHOOTER, UVES, HARPS, AMBER?
            unique_instruments = set(table['instrument'])
            print("Instruments found:", unique_instruments)
            
            # Look for 1D spectra indicators
            # We want 'dp_type' or similar if available, or just check specific instruments
            
            # Filter for XSHOOTER (wide coverage)
            xs_rows = [row for row in table if 'XSHOOTER' in row['instrument']]
            print(f"XSHOOTER entries: {len(xs_rows)}")
        else:
            print("No records found in Main.")

    except Exception as e:
        print(f"ESO Main Query Error: {e}")

def search_eso_tap():
    print("\nQuerying ESO TAP Service (Science Portal/Phase 3)...")
    try:
        eso_tap = TapPlus(url="http://archive.eso.org/tap_obs")
        
        # 1. Inspect Columns
        print("Inspecting ivoa.ObsCore columns...")
        job_cols = eso_tap.launch_job("SELECT TOP 1 * FROM ivoa.ObsCore")
        res_cols = job_cols.get_results()
        print("Columns:", res_cols.colnames)
        
        # Determine correct column names
        inst_col = 'instrument_name' if 'instrument_name' in res_cols.colnames else 'instrument'
        dp_col = 'dataproduct_type' if 'dataproduct_type' in res_cols.colnames else 'dp_type'
        
        print(f"Using columns: {inst_col}, {dp_col}")
        
        # 2. Run Query
        ra = 88.79293
        dec = 7.40706
        radius = 0.05 
        
        query = f"""
        SELECT TOP 20
            target_name, {inst_col}, {dp_col}, access_url, t_min, t_max, em_min, em_max
        FROM
            ivoa.ObsCore
        WHERE
            CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
            AND {dp_col} = 'spectrum'
        """
        # Note: 'spectrum' case sensitivity? usually 'spectrum' or 'SPECTRUM'
        
        print("Executing ADQL query...")
        job = eso_tap.launch_job(query)
        table = job.get_results()
        
        if len(table) > 0:
            print(f"Found {len(table)} Science Products!")
            row = table[0]
            url = row['access_url']
            print(f"Target: {row['target_name']}, Inst: {row[inst_col]}")
            print(f"Downloading: {url}")
            
            import requests
            from astropy.io import fits
            import tempfile
            import os
            
            # Download
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
                    for chunk in r.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = tmp.name
                
                print(f"Saved to {tmp_path}")
                try:
                    with fits.open(tmp_path) as hdul:
                        with open("d:/HELIOS/eso_info.txt", "w") as f:
                            hdul.info(output=f)
                            if len(hdul) > 1:
                                f.write("\nExtension 1 Columns:\n")
                                f.write(str(hdul[1].columns.names))
                                f.write("\nHeader Check (Unit):\n")
                                # typical ESO keys
                                keys = ['BUNIT', 'TUNIT1', 'TUNIT2']
                                for k in keys:
                                    if k in hdul[1].header:
                                        f.write(f"{k}: {hdul[1].header[k]}\n")
                                    
                except Exception as e:
                    print(f"FITS Error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                print(f"Download Error: {r.status_code}")

        else:
            print("No spectra found via TAP.")

    except Exception as e:
        print(f"ESO TAP Query Error: {e}")

if __name__ == '__main__':
    search_eso_astroquery()
    search_eso_tap()
