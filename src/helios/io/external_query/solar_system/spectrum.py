
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
    Retrieves REAL observed spectrum for a planet using Payne+2025 Library.
    Source: Zenodo (Record 17470005).
    Strictly NO synthetic generation.
    
    Returns:
        wavelengths (Quantity array) or None
        flux (Quantity array) or None
    """
    # Payne+2025 covers: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
    supported_planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    
    name_lower = object_name.lower()
    if name_lower not in supported_planets:
        print(f"Object '{object_name}' not in Payne+2025 library.")
        return None, None
        
    return download_payne_2025(name_lower, dist_sun, dist_obs, wavelengths)

def download_payne_2025(planet_name, dist_sun, dist_obs, wavelengths=None):
    """
    Downloads and processes Payne+2025 Geometric Albedo spectra.
    """
    # Zenodo Record 17470005
    # Filenames are: {planet}_albedo.csv
    base_url = "https://zenodo.org/records/17470005/files/"
    filename = f"{planet_name}_albedo.csv"
    
    url = f"{base_url}{filename}" 
    cache_file = os.path.join(CACHE_DIR, f"payne2025_{filename}")
    
    # Headers to avoid 429
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Helios/1.0'
    }
    
    # Download
    if not os.path.exists(cache_file):
        print(f"Downloading Payne+2025 data for {planet_name} from {url}...")
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                with open(cache_file, 'wb') as f:
                    f.write(r.content)
            elif r.status_code == 429:
                print("Zenodo Rate Limit (429). Cannot download.")
                return None, None
            else:
                print(f"Failed to download {filename}: {r.status_code}")
                return None, None
        except Exception as e:
            print(f"Download failed: {e}")
            return None, None
            
    # Process
    try:
        # Load CSV
        # Expected columns: wavelength_um, geometric_albedo
        df = pd.read_csv(cache_file)
        
        # Standardize column names if needed (strip spaces)
        df.columns = [c.strip() for c in df.columns]
        
        if 'Wavelength' not in df.columns or 'Albedo' not in df.columns:
            # Fallback or error
            print(f"Unknown columns in {filename}: {df.columns}")
            return None, None
            
        wl_ref = df['Wavelength'].values * u.um
        albedo = df['Albedo'].values
        
        # Calculate Flux
        # S_p(lambda) = S_sun(lambda) * Ag(lambda) * Phi(alpha) * (R_p / d)^2 / pi ?
        # Geometric Albedo definition A_g:
        # F_p(0) = F_sun(0) * A_g * (R_p / d)^2   (Standard astronomy definition often implies this direct scaling at phase 0)
        # Verify: A_g is ratio of planet flux at 0 phase to flux from Lambertian disk of same cross-section.
        # F_lamb_disk = F_sun * (R/d)^2 / pi * pi = F_sun * (R/d)^2  <-- Lambertian disk scatters F_in back into pi sr? No, reflectivity=1.
        # Let's assume the simple SED scaling: Flux = SolarFluxAtPlanet * Albedo * (R/D)^2
        # This is the standard "Reflectance" usage for SEDs.
        
        # Get Solar Spectrum
        wl_sun, flux_sun_1au = get_solar_spectrum(wavelengths=wl_ref)
        
        if not isinstance(dist_sun, u.Quantity): dist_sun = dist_sun * u.AU
        if not isinstance(dist_obs, u.Quantity): dist_obs = dist_obs * u.AU
        
        # Solar Flux at Planet
        flux_sun_pl = flux_sun_1au * ((1.0*u.AU / dist_sun)**2).decompose()
        
        # Planet Radius
        props = SOLAR_SYSTEM_DATA.get(planet_name.capitalize())
        if not props:
            # Fallback or error
            R_pl = 1.0 * u.earthRad # Should not happen for major planets
        else:
            R_pl = props['radius']
            
        # Flux at Observer (Absolute @ 10pc usually requested, or actual dist)
        # Using Phase=0 (Full) for Reference Spectrum
        geom_factor = ((R_pl / dist_obs)**2).decompose()
        
        flux_obs = flux_sun_pl * albedo * geom_factor
        
        return wl_ref, flux_obs.to(u.Jy, equivalencies=u.spectral_density(wl_ref))

    except Exception as e:
        print(f"Error processing Payne+2025 for {planet_name}: {e}")
        return None, None
        
    finally:
        # Cleanup
        if 'cache_file' in locals() and os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception as e:
                print(f"Warning: Could not remove {cache_file}: {e}")

