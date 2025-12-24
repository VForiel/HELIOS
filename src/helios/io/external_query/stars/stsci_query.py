
import os
import requests
import numpy as np
import warnings
import tempfile
import shutil
from astropy import units as u
from astropy.io import fits
from astroquery.mast import Observations


# Direct fallback URLs for primary standards if MAST query fails or is empty
# Updated based on latest search - stis_005 seems to be a common stable version
CALSPEC_DIRECT_MAP = {
    'vega': 'http://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_010.fits',
    'alpha lyr': 'http://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_010.fits',
    'sirius': 'http://ssb.stsci.edu/cdbs/calspec/sirius_stis_001.fits',
    'alpha cma': 'https://archive.stsci.edu/hlsps/reference_atlases/cdbs/calspec/sirius_stis_001.fits',
    '109 vir': 'https://archive.stsci.edu/hlsps/reference_atlases/cdbs/calspec/109vir_stis_001.fits',
    'sun': 'https://archive.stsci.edu/hlsps/reference_atlases/cdbs/calspec/sun_reference_stis_002.fits'
}

# Alternative base: archive.stsci.edu sometimes 404s, try hst.stsci.edu or standardized path
# ssb.stsci.edu seems most reliable (HTTP)
CALSPEC_MIRROR_BASE = "http://ssb.stsci.edu/cdbs/calspec/"

def query_stsci_calspec(star_name, star_data):
    """
    Attempts to retrieve a standard CALSPEC spectrum for the star from STScI via MAST.
    Uses temporary files for FITS download to keep cache clean (only JSONs persist).
    """
    print(f"Querying MAST for CALSPEC reference of {star_name}...")
    
    # Mapping common names to MAST target names if needed
    target = star_name
    if star_name.lower() == 'vega': target = 'alpha lyr'
    
    # We will use a temp file for the FITS
    temp_fd, temp_path = tempfile.mkstemp(suffix='.fits')
    os.close(temp_fd) # Close handle so we can write/read freely
    
    fits_retrieved = False
    
    try:
        # 1. Try Direct URL Fallback
        direct_urls = []
        mapped = CALSPEC_DIRECT_MAP.get(star_name.lower()) or CALSPEC_DIRECT_MAP.get(target.lower())
        if mapped:
            direct_urls.append(mapped)
            fname = os.path.basename(mapped)
            # Add mirror versions
            direct_urls.append(CALSPEC_MIRROR_BASE + fname)
            # Fallback versions
            if '005' in fname: direct_urls.append(CALSPEC_MIRROR_BASE + fname.replace('005', '003'))
            if '005' in fname: direct_urls.append(CALSPEC_MIRROR_BASE + fname.replace('005', '010'))
            if '005' in fname: direct_urls.append(CALSPEC_MIRROR_BASE + fname.replace('005', '011'))

        for url in direct_urls:
             try:
                 # print(f"Trying direct download: {url}")
                 r = requests.get(url, verify=False, timeout=5)
                 if r.status_code == 200:
                     # VALIDATE CONTENT: Must look like FITS (start with SIMPLE)
                     content = r.content[:10]
                     if b'SIMPLE' in content or b'XTENSION' in content:
                         with open(temp_path, 'wb') as f:
                             f.write(r.content)
                         fits_retrieved = True
                         print(f"Success: Downloaded standard reference from {url}")
                         break
             except Exception:
                 pass
        
        # 2. If not found via direct map, try MAST Query
        if not fits_retrieved:
            try:
                # Use simple object query (cone search) to be robust against name variations
                obs_table = Observations.query_object(target, radius="0.005 deg")
                
                if len(obs_table) > 0:
                    # Filter for probable CALSPEC candidates (HLSP or HST)
                    mask = np.isin(obs_table['obs_collection'], ['HLSP', 'HST'])
                    
                    # CRITICAL: Exclude TESS/TICA to prevent massive downloads
                    excludes = []
                    for i, row in enumerate(obs_table):
                         row_str = str(row)
                         if 'TESS' in row_str or 'TICA' in row_str:
                             excludes.append(i)
                    
                    if excludes:
                        mask[excludes] = False
                    
                    candidate_table = obs_table[mask]
                    
                    if len(candidate_table) > 0:
                        # Find products
                        products = None
                        
                        # Fallback for generic product search
                        try:
                             # Limit to top 20 candidates
                            limit_candidates = candidate_table[:20]
                            products = Observations.get_product_list(limit_candidates)
                        except Exception: 
                            pass

                        # Iterate to find best fit
                        best_product = None
                        if products:
                            candidate_rows = []
                            for row in products:
                                uri = row['dataURI'].lower()
                                if 'tica' in uri or 'tess' in uri: continue
                                if uri.endswith('.fits') and 'calspec' in uri:
                                    candidate_rows.append(row)
                            
                            if candidate_rows:
                                # Sort: prefer 'stis' -> 'mod'
                                def sort_key(row):
                                    fname = os.path.basename(row['dataURI']).lower()
                                    score = 0
                                    if 'stis' in fname: score += 100
                                    if 'mod' in fname: score += 50
                                    return score
                                
                                candidate_rows.sort(key=sort_key, reverse=True)
                                best_product = candidate_rows[0]
                        
                        if best_product:
                            filename = os.path.basename(best_product['dataURI'])
                            print(f"Downloading {filename} from MAST to temp...")
                            
                            # Download to a specialized temp folder to handle MAST structure
                            mast_temp_dir = tempfile.mkdtemp()
                            try:
                                manifest = Observations.download_products(products, productType="SCIENCE", observation_id=best_product['obs_id'], download_dir=mast_temp_dir)
                                if manifest and len(manifest) > 0:
                                    downloaded_file = manifest['Local Path'][0]
                                    # Move to our target temp path
                                    shutil.copy2(downloaded_file, temp_path)
                                    fits_retrieved = True
                            finally:
                                shutil.rmtree(mast_temp_dir, ignore_errors=True)
            
            except Exception as e:
                # print(f"MAST Query Error details: {e}") 
                pass

        # 3. Read FITS if we found one
        if fits_retrieved and os.path.exists(temp_path):
           try:
               with fits.open(temp_path) as hdul:
                    data = hdul[1].data
                    cols = data.columns.names
                    wave_col = next((c for c in cols if 'WAVELENGTH' in c.upper()), None)
                    flux_col = next((c for c in cols if 'FLUX' in c.upper()), None)
                    
                    if wave_col and flux_col:
                        wave_angstrom = data[wave_col]
                        flux_cgs = data[flux_col]
                        
                        w_q = wave_angstrom * u.Angstrom
                        f_q = flux_cgs * u.erg / (u.s * u.cm**2 * u.Angstrom)
                        f_nu = f_q.to(u.Jy, equivalencies=u.spectral_density(w_q))
                        
                        star_data['sed']['wavelength'] = w_q.to(u.micron)
                        star_data['sed']['flux'] = f_nu
                        
                        print(f"Success: Processed CALSPEC spectrum ({len(wave_angstrom)} points) for {star_name}")
                        if 'STScI CALSPEC' not in star_data['metadata']['sources']:
                            star_data['metadata']['sources'].append('STScI CALSPEC')
           except Exception as e:
                print(f"Error reading FITS: {e}")

    finally:
        # CLEANUP: Always remove the prompt temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except: pass

    return star_data
