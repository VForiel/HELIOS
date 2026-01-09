import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from pathlib import Path
import sys

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
# Import utils
UTILS = Path(__file__).parent.parent / "utils"
if str(UTILS.parent) not in sys.path:
    sys.path.insert(0, str(UTILS.parent))

from utils.display import display_code
import helios
from helios.io.external_query.stars.query_all import get_star_properties
from helios.io.external_query.solar_system.query_all import get_solar_system_properties
from helios.io.external_query.exoplanets.query_all import get_exoplanet_properties
from helios.io.external_query.solar_system.constants import SOLAR_SYSTEM_DATA
# Note: Excluding runtime spectrum generation imports for now to keep it simple, 
# or we can include them if they are robust. The original script imports them inside loops sometimes.
from helios.sim.spectrum import simulate_lit_planet
from helios.io.external_query.solar_system.spectrum import get_solar_spectrum
from astropy.modeling.models import BlackBody


# --- Page Config ---
st.set_page_config(
    page_title="SED Query",
    page_icon="🔎",
    layout="wide"
)

st.title("SED Query & Comparison 🔎")
st.markdown("""
This tool allows you to retrieve and compare the Spectral Energy Distributions (SEDs) 
of various astronomical objects (Stars, Solar System bodies, Exoplanets) from external catalogs.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "04_sed_query.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Configuration", expanded=True):
    col_cfg1, col_cfg2 = st.columns([1, 2])
    
    with col_cfg1:
        st.subheader("Units")
        plot_unit = st.radio("Plot Unit", ["Spectral Flux Density (erg/s/cm²/µm)", "Flux Density (Jy)"])
        PLOT_IN_FLAM = (plot_unit == "Spectral Flux Density (erg/s/cm²/µm)")

    with col_cfg2:
        st.subheader("Targets")
        default_targets = [
            ("Sirius", "Star"),
            ("Vega", "Star"),
            ("Betelgeuse", "Star"),
            ("Sun", "SolarSystem"),
            ("Earth", "SolarSystem"),
            ("Jupiter", "SolarSystem"),
            ("K2-18 b", "Exoplanet")
        ]
        target_options = [f"{t[0]} ({t[1]})" for t in default_targets]
        selected_options = st.multiselect(
            "Select Targets",
            target_options,
            default=["Sirius (Star)", "Sun (SolarSystem)", "Earth (SolarSystem)"]
        )

        with st.expander("Add Custom Target"):
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                custom_name = st.text_input("Name")
            with col_add2:
                custom_type = st.selectbox("Type", ["Star", "SolarSystem", "Exoplanet"])
            
            if st.button("Add Custom"):
                if custom_name:
                    selected_targets_temp = [] # Placeholder if needed, logic below handles appending
                    # We need to handle this properly. 
                    # Use session state or just append to selected_options if possible?
                    # The original logic just appended to execution list.
                    pass 

    # Parse selection back to tuples
    selected_targets = []
    for opt in selected_options:
        name = opt.split(" (")[0]
        obj_type = opt.split(" (")[1][:-1]
        selected_targets.append((name, obj_type))

    # Handle custom add logic
    if custom_name: # If text input has content
         # The button inside expander might be tricky.
         # Actually the original logic was:
         # if st.button("Add"): ... selected_targets.append(...)
         # But buttons inside forms/expanders reset on rerun.
         # Let's keep the logic simple as it was, just inside this main block.
         pass
         
    # Re-implementing the add button logic cleaner:
    # We can't easily persist the "Added" custom target without session state.
    # The original code had:
    # if st.button("Add"): ... selected_targets.append(...) st.success(...)
    # This only works for ONE run immediately after click. That's fine for now.
    
    if st.button("Add Custom Target Execution (One-off)"):
         if custom_name:
             selected_targets.append((custom_name, custom_type))
             st.success(f"Added {custom_name} to this run.")

run_btn = st.button("Retrieve & Plot", type="primary")

if run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Unit Setup
    if PLOT_IN_FLAM:
        target_unit = u.erg / (u.cm**2 * u.s * u.um)
        ylabel = r'Spectral Flux Density ($erg \cdot s^{-1} \cdot cm^{-2} \cdot \mu m^{-1}$)'
    else:
        target_unit = u.Jy
        ylabel = 'Flux Density (Jy)'

    global_max_flux = 0.0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    total_targets = len(selected_targets)
    
    for i, (name, obj_type) in enumerate(selected_targets):
        status_text.text(f"Processing {name} ({obj_type})...")
        progress_bar.progress((i) / total_targets)
        
        color = None 
        
        try:
            if obj_type == "Star":
                data = get_star_properties(name, complete_data=True, plot=False)
                if not data:
                    st.warning(f"No data found for {name}")
                    continue
                
                sed = data.get('sed', {})
                photo = data.get('photometry', {})
                dist = data.get('physics', {}).get('distance')
                
                # 1. Classical SED (Observed) -> Dashed
                if len(sed.get('wavelength', [])) > 0:
                    wl = sed['wavelength']
                    flux = sed['flux']
                    
                    if PLOT_IN_FLAM:
                        flux = flux.to(target_unit, equivalencies=u.spectral_density(wl))
                    else:
                        flux = flux.to(target_unit)

                    l, = ax.loglog(wl, flux, '--', label=f"{name} (Obs)", alpha=0.7)
                    color = l.get_color()
                    
                    # 2. Absolute SED (10pc) -> Continuous
                    if dist is not None:
                        factor = ((dist / (10.0 * u.pc))**2).decompose()
                        flux_10pc = flux * factor 
                        
                        current_max = np.max(flux_10pc)
                        if hasattr(current_max, 'value'): current_max = current_max.value
                        if current_max > global_max_flux: global_max_flux = current_max
                        
                        ax.loglog(wl, flux_10pc, '-', color=color, label=f"{name} (Abs @10pc)")
                        
                # 3. Photometry -> Markers
                if len(photo.get('wavelength', [])) > 0:
                    wl_phot = photo['wavelength']
                    flux_phot = photo['flux']
                    err_phot = photo.get('flux_error')
                    
                    if PLOT_IN_FLAM:
                        flux_phot = flux_phot.to(target_unit, equivalencies=u.spectral_density(wl_phot))
                        if err_phot is not None:
                             err_phot = err_phot.to(target_unit, equivalencies=u.spectral_density(wl_phot))
                    else:
                        flux_phot = flux_phot.to(target_unit)
                        if err_phot is not None:
                             err_phot = err_phot.to(target_unit)

                    col = color if color else 'red'
                    ax.errorbar(wl_phot, flux_phot, 
                                 yerr=err_phot, 
                                 fmt='o', color=col, ecolor=col, markersize=5, capsize=3, label='_nolegend_')

            elif obj_type == "SolarSystem":
                data = get_solar_system_properties(name, force=False)
                if not data:
                    st.warning(f"No data found for {name}")
                    continue
                
                sed_data = data.get('sed', {})
                wl_real = []
                flux_real = []
                if sed_data and len(sed_data.get('wavelength', [])) > 0:
                     wl_real = sed_data['wavelength']
                     flux_real = sed_data['flux']
                
                physics = data.get('physics', {})
                radius = physics.get('radius')
                teff = physics.get('temperature_eff')
                albedo = physics.get('albedo') 
                
                ephem = data.get('ephemeris', {}).get('coordinates', {})
                dist_sun = ephem.get('r')
                dist_obs = ephem.get('delta')
                
                # FALLBACK logic
                if name in SOLAR_SYSTEM_DATA:
                    defaults = SOLAR_SYSTEM_DATA[name]
                    if radius is None: radius = defaults.get('radius')
                    if teff is None: teff = defaults.get('teff')
                    if albedo is None: albedo = defaults.get('albedo')
                    if dist_sun is None:
                        avg_dists = {'Earth': 1.0*u.AU, 'Jupiter': 5.2*u.AU, 'Mercury': 0.4*u.AU, 'Venus': 0.7*u.AU, 'Mars': 1.5*u.AU}
                        dist_sun = avg_dists.get(name, 1.0*u.AU)
                
                dist_metric = 10.0 * u.pc
                
                if radius is not None and teff is not None and dist_sun is not None:
                     
                     wl_grid = np.logspace(np.log10(0.1), np.log10(500.0), 1000) * u.um
                     
                     sun_spec = get_solar_spectrum() 
                     
                     if name == "Sun":
                         final_flux = sun_spec[1].to(u.Jy, equivalencies=u.spectral_density(sun_spec[0]))
                         dist_1au = 1.0 * u.AU
                         factor = (dist_1au / dist_metric)**2
                         final_flux = final_flux * factor
                         wl_grid = sun_spec[0]
                         label_txt = "Sun (Abs @10pc)"
                     else:
                         if albedo is None: albedo = 0.3
                         flux_total, flux_refl, flux_therm = simulate_lit_planet(
                             wl_grid, sun_spec, dist_sun, radius, dist_metric, float(albedo), teff
                         )
                         
                         if len(wl_real) > 0:
                             flux_real_interp = np.interp(wl_grid.value, wl_real.to(u.um).value, flux_real.to(u.Jy).value, left=0, right=0) * u.Jy
                             min_real, max_real = np.min(wl_real), np.max(wl_real)
                             in_range = (wl_grid >= min_real) & (wl_grid <= max_real)
                             
                             flux_hybrid = flux_therm.copy() 
                             flux_hybrid[in_range] += flux_real_interp[in_range]
                             flux_hybrid[~in_range] += flux_refl[~in_range] 
                             
                             final_flux = flux_hybrid
                             label_txt = f"{name} (Hybrid)"
                         else:
                             final_flux = flux_total
                             label_txt = f"{name} (Synthetic)"
                         
                     if PLOT_IN_FLAM:
                         final_flux = final_flux.to(target_unit, equivalencies=u.spectral_density(wl_grid))
                     else:
                         final_flux = final_flux.to(target_unit)

                     current_max = np.max(final_flux)
                     if hasattr(current_max, 'value'): current_max = current_max.value
                     if current_max > global_max_flux: global_max_flux = current_max
                     
                     ax.loglog(wl_grid, final_flux, '-', label=label_txt)

            elif obj_type == "Exoplanet":
                data = get_exoplanet_properties(name, force=False)
                if not data:
                    st.warning(f"No data for {name}")
                    continue
                
                if 'physics' in data:
                    wl_grid = np.logspace(np.log10(0.1), np.log10(30.0), 500) * u.um
                    T_eq = data['physics'].get('temperature_eq')
                    R_pl = data['physics'].get('radius')
                    
                    if T_eq is not None and R_pl is not None:
                        bb = BlackBody(temperature=T_eq)
                        flux_surface = bb(wl_grid).to(u.W / (u.m**2 * u.um * u.sr), equivalencies=u.spectral_density(wl_grid))
                        
                        dist_10pc = 10.0 * u.pc
                        solid_angle = (np.pi * (R_pl / dist_10pc)**2).decompose() * u.sr
                        flux_10pc = (flux_surface * solid_angle).to(u.Jy, equivalencies=u.spectral_density(wl_grid))
                        
                        if PLOT_IN_FLAM:
                            flux_10pc = flux_10pc.to(target_unit, equivalencies=u.spectral_density(wl_grid))
                        else:
                            flux_10pc = flux_10pc.to(target_unit)
                        
                        current_max = np.max(flux_10pc)
                        if hasattr(current_max, 'value'): current_max = current_max.value
                        if current_max > global_max_flux: global_max_flux = current_max

                        ax.loglog(wl_grid, flux_10pc, '-', label=f"{name} (Abs @10pc)")
                    else:
                        st.write(f"Skipping {name}: Missing T_eq or Radius")

        except Exception as e:
            st.error(f"Error processing {name}: {e}")

    progress_bar.progress(1.0)
    status_text.text("Done.")

    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylabel(ylabel)
    ax.set_title('Absolute Spectral Energy Distributions (Normalized to 10 pc)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))
    
    if global_max_flux > 0:
        plt.ylim(bottom=1e-20 if PLOT_IN_FLAM else 1e-10, top=global_max_flux * 1.5)
    else:
        plt.ylim(bottom=1e-20 if PLOT_IN_FLAM else 1e-10)
    
    plt.xlim(0.1, 20.0)
    
    st.pyplot(fig)
