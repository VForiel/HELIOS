
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
        res = v.query_region(target_coord, radius=10*u.arcsec, catalog='III/202')
        
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
        
    return star_data
