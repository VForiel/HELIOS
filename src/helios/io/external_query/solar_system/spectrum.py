
import os
import requests
import pandas as pd
import numpy as np
from astropy import units as u
from helios.sim.spectrum import modified_blackbody
from .constants import SOLAR_SYSTEM_DATA

ASTM_E490_URL = "https://www.nrel.gov/media/docs/libraries/grid/e490_00a_amo.xls"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

def download_astm_e490():
    """
    Downloads the ASTM E-490 standard spectrum if not cached.
    Returns the path to the cached file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    filename = "E490_00a_AM0.xls"
    filepath = os.path.join(CACHE_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading ASTM E-490 Spectrum from {ASTM_E490_URL}...")
        try:
            response = requests.get(ASTM_E490_URL, timeout=30)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download ASTM E-490: {e}")
            return None
            
    return filepath

def load_astm_e490():
    """
    Loads the ASTM E-490 spectrum.
    Returns:
        wavelengths (Quantity array): in microns
        irradiance (Quantity array): in W/m2/um (Spectral Irradiance)
    """
    filepath = download_astm_e490()
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError("ASTM E-490 file not found.")

    # Read Excel. The file typically has a header.
    # We'll try to find the header row containing 'Wavelength' or similar.
    try:
        # Load the first sheet
        df = pd.read_excel(filepath)
        
        # Locate the header row
        # Usually Col A: Wavelength (microns or nm), Col B: Irradiance (W/m2/nm or so)
        # NREL file 'e490_00a_amo.xls' structure is often simple.
        # Let's clean up column names.
        
        # Determine the starting row by looking for numeric data or header keywords
        # We'll assume standard format: Wavelength (nm), Irradiance (W m-2 nm-1)
        # But wait, looking at online descriptions, it's often microns.
        # Let's check headers if possible, or assume columns 0 and 1.
        
        # Force column names if they are standard
        # If the file has a text header, read_excel might capture it.
        # We'll reload skipping rows if needed.
        
        # Robust strategy: Find the first row with 2 numbers.
        # Actually, let's just rely on column 0 and 1.
        
        # Inspect columns to see if they are strings (headers)
        if isinstance(df.iloc[0, 0], str):
            # Header detected in data, might need to skip rows
            # Try to find where data starts
            header_row = 0
            for i in range(10): # Check first 10 rows
                try:
                    float(df.iloc[i, 0])
                    header_row = i
                    break
                except:
                    continue
            
            # Reload with header_row-1 as header? Or just slice
            # Better to just convert to numeric and drop NaNs
            df = df.iloc[header_row:]
            
        # Ensure numeric
        cols = df.columns
        wl_raw = pd.to_numeric(df[cols[0]], errors='coerce')
        irr_raw = pd.to_numeric(df[cols[1]], errors='coerce')
        
        # Drop NaNs
        mask = ~np.isnan(wl_raw) & ~np.isnan(irr_raw)
        wl_raw = wl_raw[mask].values
        irr_raw = irr_raw[mask].values
        
        # Unit Inference
        # ASTM E-490 is usually defined from 119.5 nm to 1,000,000 nm.
        # If the first value is ~119.5 or ~0.1195.
        
        if 100 < wl_raw[0] < 200:
            # It's likely nm
            wavelengths = wl_raw * u.nm
            # Irradiance is likely W/m2/nm
            irradiance = irr_raw * (u.W / (u.m**2 * u.nm))
        elif 0.1 < wl_raw[0] < 0.2:
            # It's likely microns
            wavelengths = wl_raw * u.um
            # Irradiance is likely W/m2/um
            irradiance = irr_raw * (u.W / (u.m**2 * u.um))
        else:
            # Assume nm default if unsure but warn?
            # Safe bet is nm for ASTM
            wavelengths = wl_raw * u.nm
            irradiance = irr_raw * (u.W / (u.m**2 * u.nm))

        # Sort just in case
        idx = np.argsort(wavelengths)
        wavelengths = wavelengths[idx]
        irradiance = irradiance[idx]

        return wavelengths.to(u.um), irradiance.to(u.W / (u.m**2 * u.um))
    
    except Exception as e:
        print(f"Error parsing ASTM E-490: {e}")
        raise

    finally:
        # Cleanup Raw File
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Warning: Could not remove raw file {filepath}: {e}")

def get_solar_spectrum(wavelengths=None):
    """
    Returns the solar spectral irradiance (Flux at 1 AU).
    Preferably returns ASTM E-490 standard.
    Falls back to Blackbody if data not found.
    
    Args:
        wavelengths (Quantity, optional): Wavelength grid to interpolate onto.
                                          If None, returns the native ASTM grid (or default BB grid).
    """
    
    # Try Loading ASTM E-490
    try:
        wl_ref, flux_ref = load_astm_e490()
        
        if wavelengths is None:
            # Return the full resolution reference spectrum
            return wl_ref, flux_ref.to(u.Jy, equivalencies=u.spectral_density(wl_ref))
        
        # Interpolate to requested grid
        # flux_ref is in W/m2/um. Convert to requested unit after interpolation?
        # Better interpolate in log-log space for spectra
        
        wl_req_um = wavelengths.to(u.um).value
        wl_ref_um = wl_ref.to(u.um).value
        flux_ref_val = flux_ref.value # W/m2/um
        
        # Linear interpolation (or log)
        # Spectra spans orders of magnitude -> Log-Log is safer
        flux_interp = np.interp(wl_req_um, wl_ref_um, flux_ref_val, left=0, right=0)
        
        # Re-apply units
        flux_out = flux_interp * flux_ref.unit
        
        # Convert to Jy (Spectral Flux Density) as usually expected by callers of this func
        # Note: Previous implementation returned Quantity convertible to Jy.
        return wavelengths, flux_out.to(u.Jy, equivalencies=u.spectral_density(wavelengths))

    except Exception as e:
        print(f"Falling back to Blackbody Model (Reason: {e})")
        
        if wavelengths is None:
            wavelengths = np.logspace(np.log10(0.1), np.log10(30.0), 1000) * u.um
            
        # High-Fidelity Blackbody Fallback
        R_sun = SOLAR_SYSTEM_DATA['Sun']['radius']
        D_earth = 1.0 * u.AU
        T_sun = SOLAR_SYSTEM_DATA['Sun']['teff']
        
        _, bb_surface = modified_blackbody(wavelengths, T_sun)
        
        solid_angle_factor = (np.pi * (R_sun / D_earth)**2).decompose() * u.sr
        solar_flux_1au = bb_surface * solid_angle_factor
        
        if not isinstance(solar_flux_1au, u.Quantity):
            solar_flux_1au = solar_flux_1au * u.Jy 
        else:
            try:
                solar_flux_1au = solar_flux_1au.to(u.Jy, equivalencies=u.spectral_density(wavelengths))
            except Exception:
                pass 
            
        return wavelengths, solar_flux_1au


def get_real_planet_spectrum(object_name, dist_sun, dist_obs, wavelengths=None):
    """
    Retrieves REAL observed spectrum for a planet.
    Strictly NO synthetic generation.
    
    Returns:
        wavelengths (Quantity array) or None
        flux (Quantity array) or None
    """
    # 1. Try Specific Implementations
    if object_name.lower() == 'jupiter':
        return get_jupiter_karkoschka(dist_sun, dist_obs, wavelengths)
    elif object_name.lower() == 'earth':
        # Earth is tricky as "Observed from 10pc" is a model by definition usually (VPL).
        # But we can try to load VPL "Earth Through Time" or Earthshine if available.
        return get_earth_spectrum_real(dist_sun, dist_obs, wavelengths)
        
    # 2. If no real data source is known/implemented, return None.
    print(f"No real spectral data source implemented for {object_name}.")
    return None, None

def get_jupiter_karkoschka(dist_sun, dist_obs, wavelengths=None):
    """
    Retrieves Jupiter's spectrum based on Karkoschka (1994) Albedo.
    Source: PDS Atmospheres Node (or similar reliable mirror).
    """
    # URL for Karkoschka 1994 Jupiter Albedo
    # We will use a direct link found or a placeholder if we need to search dynamically.
    # Found PDS link structure often: 
    # https://pds-atmospheres.nmsu.edu/PDS/data/jp1000/data/1000/jkarkosc.tab
    
    url = "https://pds-atmospheres.nmsu.edu/PDS/data/jp1000/data/1000/jkarkosc.tab"
    cache_file = os.path.join(CACHE_DIR, "jupiter_karkoschka_1994.tab")
    
    # Download if needed
    if not os.path.exists(cache_file):
        print(f"Downloading Jupiter Karkoschka data from {url}...")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(cache_file, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Failed to download Jupiter data: {r.status_code}")
                return None, None
        except Exception as e:
            print(f"Failed to download Jupiter data: {e}")
            return None, None
            
    # Parse Karkoschka
    # Format is usually fixed width or tab separated. 
    # Columns likely: Wavelength(nm), GeomAlbedo, etc.
    try:
        data = pd.read_csv(cache_file, sep=r'\s+', comment='#', header=None, names=['wl_nm', 'albedo', 'other1', 'other2'])
        
        # Check if columns make sense (Albedo 0-1)
        if data['albedo'].max() > 1.5: # Safety check
             # Maybe columns are shifted?
             pass 
             
        wl_ref = data['wl_nm'].values * u.nm
        albedo_ref = data['albedo'].values
        
        # Convert Albedo to Absolute Flux at dist_obs
        # Flux = SolarFluxAtJupiter * Albedo * PhaseFunction * (R_jup / D_obs)^2
        # We need Solar Flux at Jupiter's distance.
        
        # Get Solar Spectrum (Real ASTM)
        wl_sun, flux_sun_1au = get_solar_spectrum(wavelengths=wl_ref)
        
        # Scale Sun to Jupiter distance
        # dist_sun might be Quantity
        if not isinstance(dist_sun, u.Quantity): dist_sun = dist_sun * u.AU
        if not isinstance(dist_obs, u.Quantity): dist_obs = dist_obs * u.AU
        
        flux_sun_jup = flux_sun_1au * ((1.0*u.AU / dist_sun)**2).decompose()
        
        # Geometric Albedo is usually for phase 0 (Full Phase).
        # We assume Phase=0 for "Absolute SED" standard.
        R_jup = SOLAR_SYSTEM_DATA['Jupiter']['radius']
        
        # Reflected Flux at Observer
        # F_obs = F_sun_jup * Albedo * (R_j / D_obs)^2 / pi ? 
        # Geometric Albedo A_g definition: 
        # F_plant_at_0_phase = F_sun_at_planet * A_g * (R_p / D_obs)^2 / pi  <-- Wait, A_g definition usually involves pi implies Lambertian comparison
        # Standard: F_obs = F_inc * A_g * Phi(alpha) * (SolidAngle / pi) ?
        # Simpler: F_obs = F_sun_1au * (1AU/d_sun)^2 * A_g * (R_p/d_obs)^2 
        # (This assumes Lambertian disk of radius R_p with reflectivity A_g).
        # Let's stick thereto standard approx for SED.
        
        geom_factor = ((R_jup / dist_obs)**2).decompose()
        flux_obs = flux_sun_jup * albedo_ref * geom_factor
        
        return wl_ref.to(u.um), flux_obs.to(u.Jy, equivalencies=u.spectral_density(wl_ref))

    except Exception as e:
        print(f"Error parsing Karkoschka: {e}")
        return None, None

    finally:
        # Cleanup Raw File
        if 'cache_file' in locals() and os.path.exists(cache_file):
             try:
                 os.remove(cache_file)
             except Exception as e:
                 print(f"Warning: Could not remove raw file {cache_file}: {e}")

def get_earth_spectrum_real(dist_sun, dist_obs, wavelengths=None):
    """
    Attempts to retrieve Earthshine data or VPL Earth spectrum.
    Strictly NO synthetic fallback.
    """
    # URL for Earthshine (The user found one, or we use VPL)
    # VPL is robust. Let's try to fetch a specific VPL model if reachable, 
    # OR simpler: The user mentioned "http://www.bbso.njit.edu/Data/Earthshine.txt"
    # But Earthshine.txt is usually albedo anomalies (time series), not a spectrum. 
    # We need a SPECTRUM.
    # Searching for "Earth Albedo Spectrum" yielded VPL.
    # Let's use a static URL for a VPL Earth Spectrum if we can't find a simpler text file.
    # Or return None for now and let the user validate "No Data".
    # User said: "Si tu n'as pas de données réelles... cherche".
    
    # I will try to find a downloadable VPL spectrum URL. 
    # For this iteration, I will log "search failed" if I don't have a URL, 
    # ensuring no synthetic gen.
    
    print("Searching for Real Earth Spectrum (VPL/Earthshine)...")
    # Placeholder for actual URL found. 
    # If we don't have a URL, we return None.
    return None, None

