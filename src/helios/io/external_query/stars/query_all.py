
"""
Module for accessing online astronomical catalogs to retrieve comprehensive star properties and SEDs.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from datetime import datetime
import urllib3
import ssl
import requests


# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Hack: Force all requests to be unverified
old_request = requests.Session.request
def unverified_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)
requests.Session.request = unverified_request
ssl._create_default_https_context = ssl._create_unverified_context

import os
import json
import time
import socket
# Set global timeout for all socket operations (e.g. astroquery)
socket.setdefaulttimeout(15)

from helios.io.external_query.stars import query_simbad, query_vizier, query_stsci_calspec, query_vizier_spectra
from helios.io.external_query.stars.vizier_spectra_extended import query_extended_spectra
from helios.io.external_query.stars.eso_query import query_eso_spectra
from helios.io.external_query.stars.eso_query import query_eso_spectra
from helios.io.external_query.stars.irsa_query import query_irsa_iso
from helios.io.external_query.solar_system import get_solar_system_properties
from helios.io.external_query.stars.serialization import serialize_star_data, deserialize_star_data

def get_star_properties(star_name, complete_data=False, plot=False, force=False):
    """
    Retrieves properties for a given star from various online catalogs.
    
    Parameters
    ----------
    star_name : str
         Common name of the star (e.g. "Vega", "Betelgeuse").
    complete_data : bool
         If True, fills missing physics and generates a synthetic/hybrid SED.
    plot : bool
         If True, displays a plot of the SED.
    force : bool
         If True, forces a fresh query and overwrites the cache.
         If False, attempts to load from cache if < 1 year old.
         
    Returns
    -------
    dict
         Star Data Dictionary.
    """
    
    # Setup Cache Path
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Sanitize filename
    safe_name = "".join([c for c in star_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    cache_file = os.path.join(cache_dir, f"{safe_name}.json")
    
    # -------------------------------------------------------------
    # 0. DISPATCH: Solar System
    # -------------------------------------------------------------
    solar_system_names = ['Sun', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Earth']
    if star_name.capitalize() in solar_system_names:
        print(f"Detected Solar System Object: {star_name}...")
        star_data = get_solar_system_properties(star_name, complete_data=complete_data, force=force)
        return star_data

    # -------------------------------------------------------------
    # 1. DISPATCH: Exoplanets
    # -------------------------------------------------------------
    # Check if we should route to exoplanets module
    # A. Check Exoplanet Cache First
    from helios.io.external_query.exoplanets import get_exoplanet_properties
    try:
        exo_cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exoplanets", "cache", f"{safe_name}.json")
        if os.path.exists(exo_cache_path) and not force:
             # It exists in Exo cache, so it IS an exoplanet
             return get_exoplanet_properties(star_name, complete_data=complete_data, force=force)
    except: pass
    
    # B. Heuristic: Common Exoplanet naming (ends in ' b', ' c', etc or known list)
    is_likely_planet = star_name.split(' ')[-1] in ['b', 'c', 'd', 'e', 'f', 'g']
    if is_likely_planet:
        print(f"Probable Exoplanet '{star_name}' detected. Checking Archive...")
        exo_data = get_exoplanet_properties(star_name, complete_data=complete_data, force=force)
        if exo_data:
            return exo_data
        else:
            print("Not found in Exoplanet Archive, falling back to Star Search.")

    # -------------------------------------------------------------
    # 2. DISPATCH: Stars (Standard Flow)
    # -------------------------------------------------------------
    star_data = None
    cache_valid = False
    
    # Try Loading Star Cache
    if not force and os.path.exists(cache_file):
        # Check age (1 year = 365 * 24 * 3600 seconds)
        max_age = 365 * 24 * 3600
        file_age = time.time() - os.path.getmtime(cache_file)
        
        if file_age < max_age:
            try:
                print(f"Loading '{star_name}' from cache...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    raw_json = json.load(f)
                    star_data = deserialize_star_data(raw_json)
                cache_valid = True
            except Exception as e:
                print(f"Cache load failed ({e}), re-querying.")
        else:
             print(f"Cache expired (Age: {file_age/86400:.1f} days), re-querying.")
             
    # 2. Query if needed
    if not cache_valid:
        print(f"Retrieving properties for {star_name}...")
        
        # Structure to populate
        star_data = {
            'identity': {'name': star_name, 'simbad_id': None, 'aliases': []},
            'coordinates': {'ra': None, 'dec': None, 'frame': 'icrs'},
            'kinematics': {'pm_ra': None, 'pm_dec': None, 'radial_velocity': None},
            'physics': {'parallax': None, 'distance': None, 'spectral_type': None, 'temperature_eff': None},
            'sed': {'wavelength': [], 'flux': [], 'frequency': [], 'bands': []},
            'metadata': {'sources': [], 'retrieved_at': datetime.now().isoformat()},
            # Temporary storage for raw photometry before formatting
            '_sed_temp': {'wavelength': [], 'flux': [], 'flux_error': [], 'band': []}
        }

        # Query Chain
        star_data = query_simbad(star_name, star_data)
        
        if star_data['coordinates']['ra'] is not None:
            star_data = query_vizier(star_name, star_data)
            
            # --- Spectral Data Accumulation ---
            sed_segments = []
            
            def capture_segment(s_data, source_name):
                if len(s_data['sed']['wavelength']) > 0:
                    sed_segments.append({
                        'wavelength': s_data['sed']['wavelength'],
                        'flux': s_data['sed']['flux'],
                        'source': source_name
                    })
                    # Clear for next query
                    s_data['sed']['wavelength'] = []
                    s_data['sed']['flux'] = []
                    return True
                return False

            # 1. STScI CALSPEC
            star_data = query_stsci_calspec(star_name, star_data)
            capture_segment(star_data, 'CALSPEC')
            
            # 2. ESO Archive (Phase 3)
            star_data = query_eso_spectra(star_name, star_data)
            capture_segment(star_data, 'ESO')
            
            # 3. Extended Catalogs (Alekseeva, Glushneva, etc.)
            star_data = query_extended_spectra(star_name, star_data)
            capture_segment(star_data, 'Vizier Extended')
            
            # 4. IRSA / ISO Query
            # -------------------
            # Query for ISO SWS spectrum via SSA
            irsa_segments = query_irsa_iso(star_name)
            if irsa_segments:
                sed_segments.extend(irsa_segments)
                star_data['sed'] = {'wavelength': [], 'flux': []}

            # 5. Fallback Vizier Spectra (Burnashev)
            star_data = query_vizier_spectra(star_name, star_data)
            capture_segment(star_data, 'Vizier General')

            # 6. Fallback: Theoretical Models (Pickles, etc.)
            # If we still have very poor coverage
            has_good_data = len(sed_segments) > 0
            # Heuristic: If we only have < 50 points, try models? 
            # Or always try models for comparison? 
            # Let's fetch model if coverage is low.
            
            total_points = sum([len(s['wavelength']) for s in sed_segments])
            if total_points < 100 or star_name in ["Betelgeuse", "K2-18"]: # Specific check for Demo targets
                 from helios.io.external_query.stars.model_query import get_model_sed
                 print(f"Attempting model retrieval for {star_name}...")
                 model_seg = get_model_sed(star_name, star_data)
                 if model_seg:
                     # Add as segment. Priority? 
                     # If we have NO data, this is main.
                     # If we have some data, this helps context.
                     sed_segments.append(model_seg)
                     print("  > Model added to segments.")

            # --- Merge Logic ---
            if sed_segments:
                star_data = merge_sed(star_data, sed_segments)
        
        # Process retrieved photometry into Star Data Structure
        sed_temp = star_data.pop('_sed_temp')
        
        if len(sed_temp['wavelength']) > 0:
             star_data['photometry'] = {
                'wavelength': np.array(sed_temp['wavelength']) * u.um,
                'flux': np.array(sed_temp['flux']) * u.Jy,
                'flux_error': np.array(sed_temp['flux_error']) * u.Jy,
                'bands': np.array(sed_temp['band'])
             }
        
        # Save to Cache (Raw Data)
        try:
            print(f"Saving '{star_name}' to cache...")
            serializable_data = serialize_star_data(star_data)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, separators=(',', ':'))
        except Exception as e:
            print(f"Failed to save cache: {e}")

    # 3. Data Completion (Optional)
    if complete_data:
         from helios.utils.data_completion.star import complete_star_data
         
         # Ensure Photometry structure is ready
         if len(star_data['sed']['wavelength']) == 0 and 'photometry' in star_data:
              pass
         
         star_data = complete_star_data(star_data)
         
    # 4. Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        
        # High-Res Model
        if len(star_data['sed']['wavelength']) > 0:
            val = star_data['physics'].get('temperature_eff')
            T_label = f"{val.value:.0f} K" if hasattr(val, 'value') else f"{val} K"
            plt.loglog(star_data['sed']['wavelength'], star_data['sed']['flux'], '-', 
                       label=f"Model ({T_label})", color='gray', alpha=0.7)
        
        # Photometry Points with Errors
        photo = star_data.get('photometry', {})
        if photo and len(photo.get('wavelength', [])) > 0:
            plt.errorbar(photo['wavelength'], photo['flux'], 
                         yerr=photo.get('flux_error'), 
                         fmt='o', color='red', label='Photometry', ecolor='salmon', capsize=3)
            
            if 'bands' in photo:
                for i, txt in enumerate(photo['bands']):
                     # Handle quantities for plotting
                    wl_val = photo['wavelength'][i].value if hasattr(photo['wavelength'][i], 'value') else photo['wavelength'][i]
                    flux_val = photo['flux'][i].value if hasattr(photo['flux'][i], 'value') else photo['flux'][i]
                    plt.annotate(txt, (wl_val, flux_val), 
                                 xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)

        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Flux Density (Jy)')
        # Handle spectral type safely
        sp = star_data["physics"].get("spectral_type", "Unknown")
        plt.title(f'SED: {star_name} ({sp})')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.show()

    return star_data

def merge_sed(star_data, segments):
    """
    Combines multiple spectral segments into a single composite SED.
    
    Parameters
    ----------
    star_data : dict
        Main star data structure.
    segments : list of dict
        List of SED segments, each being a dict {'wavelength': [], 'flux': [], 'source': str}.
        
    Returns
    -------
    dict (star_data)
        Updated star_data with merged SED.
    """
    if not segments:
        return star_data
        
    print(f"Merging {len(segments)} spectral segments...")
    
    # 1. Collect all points
    all_wave = []
    all_flux = []
    
    # Simple strategy: Concatenate and Sort
    # Advanced strategy: Resample to finest grid? No, just keep all points for now.
    # Overlap strategy: Higher priority sources came first?
    # Actually, let's assume segments are passed in priority order? 
    # Or just simpler: Sort by wavelength.
    
    # Strategy: Just concatenate all. If overlaps exist, we might have double points.
    # Let's simple-sort for now as a robust start.
    
    for seg in segments:
        w = seg['wavelength']
        f = seg['flux']
        
        # Ensure units
        if not hasattr(w, 'unit'): w = w * u.micron
        if not hasattr(f, 'unit'): f = f * u.Jy
        
        all_wave.append(w.to(u.micron).value)
        all_flux.append(f.to(u.Jy).value)
        
    # Flatten
    if len(all_wave) > 0:
        flat_wave = np.concatenate(all_wave)
        flat_flux = np.concatenate(all_flux)
        
        # Sort
        idx = np.argsort(flat_wave)
        final_wave = flat_wave[idx] * u.micron
        final_flux = flat_flux[idx] * u.Jy
        
        star_data['sed']['wavelength'] = final_wave
        star_data['sed']['flux'] = final_flux
        
    return star_data
