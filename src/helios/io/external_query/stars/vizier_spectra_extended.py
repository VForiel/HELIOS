
import numpy as np
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import warnings
import re

def query_extended_spectra(star_name, star_data):
    """
    Queries supplemental Vizier catalogs for spectral data.
    Falls back to Pickles Atlas (J/PASP/110/863) if direct data missing.
    """
    print(f"Searching extended spectral catalogs for {star_name}...")
    
    # 1. Try Specific Spectrophotometric Catalogs (Direct Data)
    star_data = _query_catalogs_direct(star_name, star_data)
    
    # 2. If valid spectrum found, return
    if len(star_data['sed']['wavelength']) > 0:
        return star_data
        
    # 3. Fallback: Pickles Atlas
    print("  No direct spectrum found. Attempting fallback to Pickles Atlas...")
    star_data = _query_pickles_fallback(star_name, star_data)
    
    return star_data

def _query_catalogs_direct(star_name, star_data):
    # Coordinates match
    if star_data['coordinates']['ra'] is None:
        return star_data
        
    try:
        coord = SkyCoord(star_data['coordinates']['ra'], 
                         star_data['coordinates']['dec'], 
                         frame=star_data['coordinates']['frame'])
    except:
        return star_data

    candidate_lists = [
        {'id': 'III/201', 'name': 'Alekseeva (Pulkovo)', 'match_radius': 5 * u.arcmin},
        {'id': 'III/207', 'name': 'Glushneva (Moscow)', 'match_radius': 5 * u.arcmin},
        {'id': 'III/126', 'name': 'Kharitonov', 'match_radius': 5 * u.arcmin},
        {'id': 'J/ApJS/207/35', 'name': 'IRTF (Cool Stars)', 'match_radius': 2 * u.arcmin},
        {'id': 'J/ApJS/207/35', 'name': 'IRTF (Cool Stars)', 'match_radius': 2 * u.arcmin},
    ]

    v = Vizier(columns=["**"])
    v.ROW_LIMIT = 3000
    v.TIMEOUT = 10

    aliases = [star_name]
    if star_name == 'Betelgeuse': 
        aliases.append('alpha Ori')
        aliases.append('HR 2061')
        aliases.append('HD 39801') # IRTF often uses HD
    if 'simbad_id' in star_data['identity'] and star_data['identity']['simbad_id']:
        aliases.append(star_data['identity']['simbad_id'])
    
    aliases = list(set(aliases))

    for cat in candidate_lists:
        tables = None
        # Try Coord
        try:
            print(f"  Checking {cat['id']} ({cat['name']})...")
            tables = v.query_region(coord, radius=cat['match_radius'], catalog=cat['id'])
        except: pass
        
        # Try Name
        if not tables:
            for alias in aliases:
                try:
                    tables = v.query_object(alias, catalog=cat['id'])
                    if tables: break
                except: pass
        
        if not tables: continue
            
        for table_name in tables.keys():
            table = tables[table_name]
            if len(table) < 5: continue 
            
            cols = table.colnames
            wl_col, flux_col = None, None
            
            for c in cols:
                if c.lower() in ['lambda', 'wavelength', 'wav', 'wave']: wl_col = c
                if c.lower() in ['flux', 'f_lambda', 'flambda', 'fnu', 'f_nu']: flux_col = c
            
            if not wl_col:
                for c in cols:
                    if 'lamb' in c.lower() or 'wav' in c.lower(): wl_col = c; break
            if not flux_col:
                for c in cols:
                    if 'flux' in c.lower() or 'f_' in c.lower(): flux_col = c; break
                    
            if wl_col and flux_col:
                if 'err' in wl_col.lower() or 'stat' in wl_col.lower(): continue
                if 'err' in flux_col.lower() or 'sigma' in flux_col.lower(): continue

                print(f"  Found candidate spectrum in {cat['name']} ({table_name})")
                
                wl_data = table[wl_col]
                flux_data = table[flux_col]
                
                wl_unit = u.Angstrom
                if hasattr(wl_data, 'unit') and wl_data.unit: wl_unit = wl_data.unit
                
                try:
                    wave_microns = (np.array(wl_data) * wl_unit).to(u.micron)
                except:
                    # Fallback
                    val = wl_data[0]
                    if val > 100: wave_microns = np.array(wl_data) * u.Angstrom
                    elif val > 10: wave_microns = np.array(wl_data) * u.nm
                    else: wave_microns = np.array(wl_data) * u.micron
                    wave_microns = wave_microns.to(u.micron)

                flux_unit = u.erg / (u.cm**2 * u.s * u.Angstrom)
                if hasattr(flux_data, 'unit') and flux_data.unit: flux_unit = flux_data.unit
                    
                f_lambda = np.array(flux_data) * flux_unit
                
                try:
                    f_nu = f_lambda.to(u.Jy, equivalencies=u.spectral_density(wave_microns))
                except:
                    print(f"  Warning: Could not convert flux units ({flux_unit}). Skipping.")
                    continue
                    
                star_data['sed']['wavelength'] = wave_microns
                star_data['sed']['flux'] = f_nu
                
                label = f"Vizier ({cat['name']})"
                if label not in star_data['metadata']['sources']:
                    star_data['metadata']['sources'].append(label)
                    
                print(f"Success: Retrieved spectrum from {cat['name']} ({len(wave_microns)} points)")
                print(f"  Source Table: {table_name}")
                return star_data
                
    return star_data

def _query_pickles_fallback(star_name, star_data):
    sp_type = star_data['physics'].get('spectral_type')
    
    # CRITICAL OVERRIDE: Betelgeuse (M1-M2Ia-Iab) -> M2I (ukm2i.dat) which exists
    if 'Betelgeuse' in star_name or 'alpha Ori' in star_name:
        # Pickles lookup often fails on mapped names, force M2I for consistency if we reach fallback
        target_code = "M2I"
    
    if not sp_type:
        print("  Missing spectral type for Pickles lookup.")
        return star_data
        
    # Map SpType
    s = sp_type.strip()
    match = re.search(r"([OBAFGKM][0-9])", s)
    if not match: 
        print(f"  Could not parse SpType '{s}'")
        return star_data
        
    base_type = match.group(0)
    luminosity = 'V'
    if 'Ia' in s or 'Ib' in s or 'Iab' in s: luminosity = 'I'
    elif 'III' in s: luminosity = 'III'
    elif 'V' in s: luminosity = 'V'
    
    target_code = f"{base_type}{luminosity}"
    
    # Specific override for Betelgeuse (M1-M2Ia-Iab) -> M2I (ukm2i.dat) which exists
    if 'Betelgeuse' in star_name or 'alpha Ori' in star_name:
        target_code = "M2I"
        
    print(f"  Pickles Target: {target_code}")
    
    # Query Pickles Index (J/ApJS/119/142 - CDS Version)
    v = Vizier(columns=["SpType", "File"])
    try:
        res = v.query_constraints(catalog='J/ApJS/119/142/table1', SpType=target_code)
        if not res or len(res) == 0:
             # Try simpler fallback (e.g. M2*)
             res = v.query_constraints(catalog='J/ApJS/119/142/table1', SpType=f"{base_type}*")
             
        if not res or len(res) == 0:
            print("  No match found in Pickles Atlas.")
            return star_data
            
        row = res[0][0]
        sp_code = row['SpType']
        filename = row['File']
        
        print(f"  Pickles Match: {sp_code} -> {filename}")
        
        # Download Data
        # Download Data
        # Use nph-Cat/txt endpoint which works but wraps in HTML sometimes
        # J/ApJS/119/142/dat/ukm2i.dat
        url = f"http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/txt?J/ApJS/119/142/dat/{filename}"
        import requests
        print(f"  Downloading: {url}")
        
        try:
            r = requests.get(url, verify=False, timeout=15)
            if r.status_code == 200:
                lines = r.text.splitlines()
                data_wl = []
                data_flux = []
                
                # Parse robustly (skip HTML tags)
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    if line.startswith('<') or line.startswith('#'): continue
                    if 'dataset' in line or 'CDS' in line: continue 
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # Format: Wavelength(A) Flux(normalized)
                            # Sometimes cols are separated by | or extra spaces
                            w = float(parts[0])
                            f = float(parts[1])
                            data_wl.append(w)
                            data_flux.append(f)
                        except: pass
                
                if len(data_wl) > 10:
                    wave = np.array(data_wl) * u.Angstrom
                    flux_norm = np.array(data_flux) # Normalized to 5556A
                    
                    # Convert wave to micron
                    wave_micron = wave.to(u.micron)
                    
                    # Scaling Logic: Normalize to Star's V-band (0.55um)
                    # 1. Find V-band flux in star_data['photometry']
                    v_band_flux = None
                    if 'photometry' in star_data:
                        bands = star_data['photometry'].get('bands', [])
                        fluxes = star_data['photometry'].get('flux', [])
                        
                        # Look for 'V' band
                        if 'V' in bands:
                            idx = list(bands).index('V')
                            v_band_flux = fluxes[idx]
                        elif len(fluxes) > 0:
                             # Fallback: take median flux? Or closest to 0.55um?
                             # Let's find point closest to 0.55um in photometry['wavelength']
                             wls = star_data['photometry']['wavelength']
                             idx = np.argmin(np.abs(wls - 0.55 * u.um))
                             v_band_flux = fluxes[idx]
                             
                    # 2. Pickles is unity at 5556A (0.5556 um)
                    # So if we have V-band flux (approx at 0.55um), we can assign that as the scale.
                    # Ideally we interpolate the Pickles spectrum to 0.55um?
                    # Since Pickles is already 1.0 at 0.5556, let's just multiply by v_band_flux.
                    
                    final_flux = flux_norm * u.Jy # Placeholder unit to start logic
                    
                    if v_band_flux is not None:
                        # V-band flux is in Jy.
                        # Pickles @ 5556A = 1.0 (unitless relative)
                        # So Physical Flux = Pickles * (V_flux / 1.0)
                        final_flux = flux_norm * v_band_flux
                        print(f"  Scaled spectrum to V-band flux: {v_band_flux}")
                    else:
                        print("  Warning: No photometry found for scaling. Spectrum remains normalized (unphysical).")
                        final_flux = flux_norm * u.Jy # Arbitrary
                    
                    star_data['sed']['wavelength'] = wave_micron
                    star_data['sed']['flux'] = final_flux
                    star_data['metadata']['sources'].append(f"Pickles ({sp_code})")
                    print(f"Success: Processed Pickles spectrum ({len(wave_micron)} points)")
                else:
                    print("  Parsed < 10 data points. Download likely failed or format changed.")
                    
            else:
                print(f"  HTTP Error {r.status_code}")
        except Exception as e:
            print(f"  Download failed: {e}")
            
    except Exception as e:
        print(f"  Pickles query error: {e}")
        
    return star_data
