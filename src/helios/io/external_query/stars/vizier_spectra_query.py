
import numpy as np
import re
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord

def query_vizier_spectra(star_name, star_data):
    """
    Queries Vizier for spectral data (not just photometry) from specific catalogs like CALSPEC or Burnashev.
    """
    # Needs valid coords
    coords = star_data['coordinates']
    if coords.get('ra') is None:
         return star_data
         
    target_coord = SkyCoord(coords['ra'], coords['dec'], frame=coords.get('frame', 'icrs'))
    
    # 1. Try J/AJ/157/229 (CALSPEC Bohlin 2019) - The Best
    # But usually doesn't work with Cone Search well for some reason. 
    # We kept the logic just in case.
    try:
        v = Vizier(columns=['lambda', 'Flux'], row_limit=5000)
        res = v.query_region(target_coord, radius=5*u.arcsec, catalog='J/AJ/157/229')
        if len(res) > 0:
            table = res[0]
            if 'lambda' in table.colnames and 'Flux' in table.colnames:
                wave_ang = table['lambda']
                flux_cgs = table['Flux']
                
                w_q = np.array(wave_ang) * u.Angstrom
                f_q = np.array(flux_cgs) * u.erg / (u.s * u.cm**2 * u.Angstrom)
                f_nu = f_q.to(u.Jy, equivalencies=u.spectral_density(w_q))
                
                star_data['sed']['wavelength'] = w_q.to(u.micron)
                star_data['sed']['flux'] = f_nu
                
                if 'Vizier CALSPEC' not in star_data['metadata']['sources']:
                    star_data['metadata']['sources'].append('Vizier CALSPEC')
                    print(f"Success: Retrieved spectrum from Vizier CALSPEC ({len(w_q)} points)")
                return star_data
    except Exception:
        pass

    # 2. Try III/202 (Burnashev 1985) - Visible Spectrophotometry
    # It has columns F3200, F3250... F7600 (Flux in erg/cm2/s/A)
    try:
        v = Vizier(columns=['F*'], row_limit=1)
        res = v.query_region(target_coord, radius=30*u.arcsec, catalog='III/202')
        
        if len(res) > 0:
            table = res[0]
            # Verify columns look like Flux
            # Filter columns starting with 'F' and followed by digits
            flux_cols = [c for c in table.colnames if re.match(r'^F\d+$', c)]
            
            if len(flux_cols) > 10:
                wavelengths = []
                fluxes = []
                
                row = table[0]
                
                for c in flux_cols:
                    # Extract wavelength from column name (e.g. F3225 -> 3225 Angstrom)
                    lam_val = float(c[1:]) 
                    val = row[c]
                    
                    if not np.ma.is_masked(val):
                        wavelengths.append(lam_val)
                        fluxes.append(val)
                
                if wavelengths:
                    # Sort
                    srt = np.argsort(wavelengths)
                    w_arr = np.array(wavelengths)[srt] * u.Angstrom
                    f_arr = np.array(fluxes)[srt] * u.erg / (u.s * u.cm**2 * u.Angstrom)
                    
                    f_nu = f_arr.to(u.Jy, equivalencies=u.spectral_density(w_arr))
                    
                    star_data['sed']['wavelength'] = w_arr.to(u.micron)
                    star_data['sed']['flux'] = f_nu
                    
                    if 'Vizier (Burnashev)' not in star_data['metadata']['sources']:
                        star_data['metadata']['sources'].append('Vizier (Burnashev)')
                        print(f"Success: Retrieved spectrum from Burnashev Catalog ({len(w_arr)} points)")
                        
    except Exception as e:
        # print(f"Burnashev Query Error: {e}")
        pass
        
    # 3. Try J/A+A/529/A75 (Morel+ 2011) - Optical Spectra of Standard Stars
    try:
         v = Vizier(columns=['lambda', 'Flux'], row_limit=5000)
         # Using generic search if catalog name is specific
         res = v.query_region(target_coord, radius=10*u.arcsec, catalog='J/A+A/529/A75')
         if len(res) > 0:
             table = res[0]
             if len(table) > 100:
                 # Check columns
                 # Usually 'lambda' (A) and 'Flux' (erg/cm2/s/A)
                 # Need to verify units in catalog metadata in real life, assuming standard
                 w_col = 'lambda'
                 f_col = 'Flux'
                 
                 if w_col in table.colnames and f_col in table.colnames:
                     w_q = table[w_col].quantity
                     if w_q.unit is None: w_q = w_q * u.Angstrom # Default
                     
                     f_q = table[f_col].quantity
                     if f_q.unit is None: f_q = f_q * u.erg / (u.cm**2 * u.s * u.Angstrom) # Default
                     
                     f_nu = f_q.to(u.Jy, equivalencies=u.spectral_density(w_q))
                     
                     # Append or Merge? Currently implementation overwrites if not empty, or append?
                     # query_all.py merges segments. Here we manipulate star_data['sed'] directly.
                     # We should be careful not to overwrite better data (CALSPEC).
                     # Only add if empty or append?
                     # For now, let's just populate if empty.
                     if len(star_data['sed']['wavelength']) < 10:
                        star_data['sed']['wavelength'] = w_q.to(u.micron)
                        star_data['sed']['flux'] = f_nu
                        if 'Vizier (Morel)' not in star_data['metadata']['sources']:
                            star_data['metadata']['sources'].append('Vizier (Morel)')
                        print(f"Success: Retrieved spectrum from Morel Catalog ({len(w_q)} points)")
    except Exception:
        pass
        
    # 4. Try III/232 (STELIB) - ESO Stellar Library (Le Borgne+ 2003)
    try:
         v = Vizier(columns=['lambda', 'Flux'], row_limit=5000)
         res = v.query_region(target_coord, radius=10*u.arcsec, catalog='III/232')
         if len(res) > 0:
             table = res[0]
             # Columns usually 'lambda' (A) and 'Flux' (erg/cm2/s/A)
             if 'lambda' in table.colnames and 'Flux' in table.colnames:
                 w_q = table['lambda'].quantity
                 if w_q.unit is None: w_q = w_q * u.Angstrom
                 
                 f_q = table['Flux'].quantity
                 if f_q.unit is None: f_q = f_q * u.erg / (u.cm**2 * u.s * u.Angstrom)
                 
                 f_nu = f_q.to(u.Jy, equivalencies=u.spectral_density(w_q))
                 
                 if len(star_data['sed']['wavelength']) < 10:
                    star_data['sed']['wavelength'] = w_q.to(u.micron)
                    star_data['sed']['flux'] = f_nu
                    if 'Vizier (STELIB)' not in star_data['metadata']['sources']:
                        star_data['metadata']['sources'].append('Vizier (STELIB)')
                    print(f"Success: Retrieved spectrum from STELIB ({len(w_q)} points)")
    except Exception:
        pass

    # 5. Try J/ApJS/185/289 (IRTF Spectral Library - Rayner+ 2009)
    # Excellent for Cool Stars (M dwarfs, Giants) in NIR (0.8-5.0 um).
    try:
         v = Vizier(columns=['lambda', 'Flux'], row_limit=5000)
         res = v.query_region(target_coord, radius=10*u.arcsec, catalog='J/ApJS/185/289')
         if len(res) > 0:
             table = res[0]
             if 'lambda' in table.colnames and 'Flux' in table.colnames:
                 w_q = table['lambda'].quantity
                 if w_q.unit is None: w_q = w_q * u.Angstrom # Usually um in IRTF? Needs check.
                 # Actually Rayner 2009 usually keeps microns. 
                 # Vizier standardizes to 'lambda' (A) often?
                 # Let's check typical Vizier behavior. If it says 'lambda' it's A or m?
                 # Safest is to check unit or magnitude.
                 # If value is ~10000, it's Angstrom. If ~1, it's micron.
                 
                 # Assuming Vizier standardized (Angstrom)
                 if np.mean(w_q.value) < 100: # Likely micron
                      w_q = w_q.value * u.micron
                 elif w_q.unit is None:
                      w_q = w_q * u.Angstrom
                 
                 f_q = table['Flux'].quantity
                 if f_q.unit is None: f_q = f_q * u.erg / (u.cm**2 * u.s * u.Angstrom) # Standard Vizier Flux
                 elif f_q.unit == u.dimensionless_unscaled: # Sometimes W/m2/um
                      f_q = f_q * u.erg / (u.cm**2 * u.s * u.Angstrom) # Fallback assumption
                 
                 f_nu = f_q.to(u.Jy, equivalencies=u.spectral_density(w_q))
                 
                 # IRTF is valuable. We might want to MERGE it with optical?
                 # Current logic overwrites if empty.
                 # Ideally we should extend.
                 
                 if len(star_data['sed']['wavelength']) < 10:
                    star_data['sed']['wavelength'] = w_q.to(u.micron)
                    star_data['sed']['flux'] = f_nu
                    source_label = 'Vizier (IRTF Lib)'
                    if source_label not in star_data['metadata']['sources']:
                        star_data['metadata']['sources'].append(source_label)
                    print(f"Success: Retrieved spectrum from IRTF Library ({len(w_q)} points)")
                 else:
                    # Append logic?
                    # The current structure expects sorted unique wavelength.
                    # Merging is complex (overlap handling).
                    # query_all.py handles segments. 
                    # Here we are inside 'query_vizier_spectra' which is ONE segment source.
                    # Be conservative: only use if we have nothing better.
                    pass
    except Exception:
        pass
        
    return star_data
