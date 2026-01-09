
"""
04_sed_query.py

Demo script to show spectrum retrieval from different catalogs
"""
import matplotlib.pyplot as plt
import os
import pprint
from helios.io.external_query.stars.query_all import get_star_properties
from helios.io.external_query.solar_system.query_all import get_solar_system_properties
from helios.io.external_query.exoplanets.query_all import get_exoplanet_properties
from astropy import units as u
import numpy as np

# Import Exoplanet Spectrum Generator for Runtime Calculation
from helios.io.external_query.exoplanets.spectrum import generate_exoplanet_spectrum
from helios.io.external_query.solar_system.constants import SOLAR_SYSTEM_DATA

def run_demo(save=False):
    
    # --- CONFIGURATION ---
    PLOT_IN_FLAM = True # If True, plot in erg/s/cm^2/um. If False, plot in Jansky.
    # ---------------------

    # Define targets
    targets = [
        ("Sirius", "Star"),
        ("Vega", "Star"),
        ("Betelgeuse", "Star"),
        ("K2-18", "Star"),
        ("Sun", "SolarSystem"),
        ("Earth", "SolarSystem"),
        ("Jupiter", "SolarSystem"),
        ("K2-18 b", "Exoplanet")
    ]
    
    plt.figure(figsize=(12, 8))
    
    # Unit Setup
    if PLOT_IN_FLAM:
        target_unit = u.erg / (u.cm**2 * u.s * u.um)
        ylabel = r'Spectral Flux Density ($erg \cdot s^{-1} \cdot cm^{-2} \cdot \mu m^{-1}$)'
    else:
        target_unit = u.Jy
        ylabel = 'Flux Density (Jy)'

    global_max_flux = 0.0
    
    for name, obj_type in targets:
        print(f"\n--- Processing {name} ({obj_type}) ---")
        color = None
        
        try:
            if obj_type == "Star":
                data = get_star_properties(name, complete_data=True, plot=False)
                if not data: continue
                
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

                    l, = plt.loglog(wl, flux, '--', label=f"{name} (Obs)", alpha=0.7)
                    color = l.get_color()
                    
                    # 2. Absolute SED (10pc) -> Continuous
                    if dist is not None:
                        factor = ((dist / (10.0 * u.pc))**2).decompose()
                        flux_10pc = flux * factor 
                        
                        current_max = np.max(flux_10pc)
                        if hasattr(current_max, 'value'): current_max = current_max.value
                        if current_max > global_max_flux: global_max_flux = current_max
                        
                        plt.loglog(wl, flux_10pc, '-', color=color, label=f"{name} (Abs @10pc)")
                        
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
                    plt.errorbar(wl_phot, flux_phot, 
                                 yerr=err_phot, 
                                 fmt='o', color=col, ecolor=col, markersize=5, capsize=3, label='_nolegend_')

            elif obj_type == "SolarSystem":
                data = get_solar_system_properties(name, force=False)
                if not data: continue
                
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
                     from helios.sim.spectrum import simulate_lit_planet
                     from helios.io.external_query.solar_system.spectrum import get_solar_spectrum
                     
                     wl_grid = np.logspace(np.log10(0.1), np.log10(500.0), 1000) * u.um
                     sun_spec = get_solar_spectrum() 
                     
                     if name == "Sun":
                         final_flux = sun_spec[1].to(u.Jy, equivalencies=u.spectral_density(sun_spec[0]))
                         dist_1au = 1.0 * u.AU
                         factor = (dist_1au / dist_metric)**2
                         final_flux = final_flux * factor
                         wl_grid = sun_spec[0]
                         label_txt = "Sun (Abs @10pc)"
                         flux_total = final_flux
                         wl_real = [] 
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
                         label_txt = f"{name} (Hybrid: Real+Synth)"
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
                     
                     print(f"  > Plotting {label_txt}. Max: {current_max:.2e}")
                     plt.loglog(wl_grid, final_flux, '-', label=label_txt)
                     
                else:
                    print(f"  > Missing physics for {name}, cannot simulate.")


            elif obj_type == "Exoplanet":
                data = get_exoplanet_properties(name, force=False)
                if not data: continue
                
                if 'physics' in data:
                    wl_grid = np.logspace(np.log10(0.1), np.log10(30.0), 500) * u.um
                    T_eq = data['physics'].get('temperature_eq')
                    R_pl = data['physics'].get('radius')
                    
                    if T_eq is not None and R_pl is not None:
                        from astropy.modeling.models import BlackBody
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

                        print(f"  > Generated Exo SED. Max Flux: {current_max}")
                        plt.loglog(wl_grid, flux_10pc, '-', label=f"{name} (Abs @10pc)")
                    else:
                        print(f"Skipping {name}: Missing T_eq or Radius")

        except Exception as e:
            print(f"Error processing {name}: {e}")

    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(ylabel)
    plt.title('Absolute Spectral Energy Distributions (Normalized to 10 pc)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))
    
    if global_max_flux > 0:
        plt.ylim(bottom=1e-20 if PLOT_IN_FLAM else 1e-10, top=global_max_flux * 1.5)
    else:
        plt.ylim(bottom=1e-20 if PLOT_IN_FLAM else 1e-10)
    
    plt.xlim(0.1, 20.0)
    plt.tight_layout()
    
    if save:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated'))
        os.makedirs(output_dir, exist_ok=True)
        filename = "04_sed_query.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    run_demo()
