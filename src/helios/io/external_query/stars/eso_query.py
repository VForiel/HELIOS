import numpy as np
from astropy import units as u
from astroquery.utils.tap.core import TapPlus
from astropy.io import fits
import requests
import tempfile
import os
import warnings

def query_eso_spectra(star_name, star_data):
    """
    Queries ESO Archive (Phase 3 Science Products) via TAP.
    Looking for 1D Spectra (XSHOOTER, UVES, etc).
    """
    print(f"Searching ESO Archive for {star_name}...")
    
    # 1. Resolve Coordinates
    ra = star_data['coordinates']['ra']
    dec = star_data['coordinates']['dec']
    if ra is None or dec is None:
        print("  Missing coordinates, skipping ESO.")
        return star_data
        
    ra_val = ra.to(u.deg).value
    dec_val = dec.to(u.deg).value
    radius = 0.05 # ~3 arcmin
    
    # 2. TAP Query
    try:
        tap = TapPlus(url="http://archive.eso.org/tap_obs")
        
        # We need to handle column name variations if they changed, 
        # but 'instrument_name' and 'dataproduct_type' are standard ObsCore.
        # We query for 'spectrum' products.
        
        query = f"""
        SELECT TOP 10
            target_name, instrument_name, dataproduct_type, access_url, em_min, em_max
        FROM
            ivoa.ObsCore
        WHERE
            CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra_val}, {dec_val}, {radius})) = 1
            AND dataproduct_type = 'spectrum'
        ORDER BY t_exptime DESC
        """
        # Ordered by exposure time to get best signal? or sorting by something else.
        
        job = tap.launch_job(query)
        table = job.get_results()
        
        if len(table) == 0:
            print("  No spectra found in ESO Archive.")
            return star_data
            
        print(f"  Found {len(table)} candidates. Checking for FITS...")
        
        # 3. Iterate candidates and try to download/parse
        for row in table:
            url = row['access_url']
            inst = row['instrument_name']
            print(f"  Checking {inst} spectrum: {url}")
            
            # Filter? XSHOOTER is best (high coverage from UV to NIR)
            # If we have XSHOOTER, prioritize it?
            # For now, take first readable one.
            
            if _process_eso_fits(url, star_data, f"ESO ({inst})"):
                return star_data
                
    except Exception as e:
        print(f"  ESO Query Error: {e}")
        
    return star_data

def _process_eso_fits(url, star_data, source_label):
    try:
        # Download temp
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code != 200:
            return False
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
            
        try:
            with fits.open(tmp_path) as hdul:
                # Expecting Phase 3 data in Extension 1 (Binary Table)
                if len(hdul) < 2:
                    return False
                    
                data = hdul[1].data
                header = hdul[1].header
                cols = hdul[1].columns.names
                
                # Identify Columns
                wl_col = None
                flux_col = None
                
                # Flexible matching
                candidates_wl = ['WAVE', 'WAVELENGTH', 'LAMBDA']
                candidates_flux = ['FLUX', 'FLUX_REDUCED', 'DATA']
                
                for c in cols:
                    if c.upper() in candidates_wl: wl_col = c
                    if c.upper() in candidates_flux: flux_col = c
                
                if not wl_col or not flux_col:
                    return False
                
                wl = data[wl_col]
                flux = data[flux_col]
                
                # Check Units
                # TUNITn in header
                # Typically WAVE is in nm or Angstrom
                # FLUX is in erg/cm2/s/A or similar
                
                # Robust Unit parsing?
                # Or try to read from header 'TUNIT1' etc based on column index?
                # This is hard to do generically without astropy table.
                
                # Let's assume standard ESO Phase 3:
                # WAVE: nm (usually)
                # FLUX: 10^-16 erg cm-2 s-1 A-1 ??
                
                # Let's try to infer from values if metadata missing
                # Betelgeuse is bright.
                
                # Use Astropy Table for easier unit handling if possible
                from astropy.table import Table
                t = Table(data)
                
                wl_unit = u.nm # Default guess
                if t[wl_col].unit:
                    wl_unit = t[wl_col].unit
                
                flux_unit = u.erg / (u.cm**2 * u.s * u.Angstrom) # Default guess
                if t[flux_col].unit:
                    flux_unit = t[flux_col].unit
                
                # Arrays
                wave_q = t[wl_col].quantity
                if not hasattr(wave_q, 'unit') or wave_q.unit is None:
                     wave_q = t[wl_col] * wl_unit
                
                flux_q = t[flux_col].quantity
                if not hasattr(flux_q, 'unit') or flux_q.unit is None:
                     flux_q = t[flux_col] * flux_unit
                     
                # Convert to internal units (micron, Jy)
                wave_micron = wave_q.to(u.micron)
                
                # Flux conversion needs spectral equivalency
                try:
                    flux_jy = flux_q.to(u.Jy, equivalencies=u.spectral_density(wave_micron))
                except:
                    # Maybe flux unit is weird?
                    print(f"    Could not convert flux unit {flux_unit} to Jy.")
                    return False
                
                # Store
                star_data['sed']['wavelength'] = wave_micron
                star_data['sed']['flux'] = flux_jy
                star_data['metadata']['sources'].append(source_label)
                
                print(f"  Success: Parsed {source_label} ({len(wave_micron)} points)")
                return True
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"  FITS Process Error: {e}")
        return False
    
    return False
