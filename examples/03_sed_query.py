
"""
Demo script to show spectrum  retrieval from different catalogs

This script will retrieve and display:
- SED of Sirius,  Vega and Betelgeuse
- Absolute SED of Sirius,  Vega and Betelgeuse (@10pc)
- Absolute SED of Sun, Earth and Jupiter (@10pc)
- Absolute SED of K2-18 b (@10pc)

Planets SED are considered using a phase angle of 0 (maximum reflection)

Each object will have it's own color.
Absolute SED are displayered in continuous line.
Classical SED are displayed in dashed line.
Rebuilt spectrum from blackbody model are displayed in dotted line.
Photometric data points are displayed as markers with error bars.
"""

import matplotlib.pyplot as plt
import pprint
from helios.io.external_query.stars.query_all import get_star_properties
from helios.io.external_query.solar_system.query_all import get_solar_system_properties
from helios.io.external_query.exoplanets.query_all import get_exoplanet_properties

def main():
    
    # Define targets
    # (Name, Type) tuples could be useful, or just distinct lists.
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
    
    from astropy import units as u
    import numpy as np
    
    # Import Exoplanet Spectrum Generator for Runtime Calculation
    from helios.io.external_query.exoplanets.spectrum import generate_exoplanet_spectrum
    
    global_max_flux = 0.0
    
    for name, obj_type in targets:
        print(f"\n--- Processing {name} ({obj_type}) ---")
        color = None # Let matplotlib cycle
        
        try:
            if obj_type == "Star":
                data = get_star_properties(name, complete_data=True, plot=False)
                if not data: continue
                
                sed = data.get('sed', {})
                photo = data.get('photometry', {})
                dist = data.get('physics', {}).get('distance')
                
                # 1. Classical SED (Observed) -> Dashed
                if len(sed.get('wavelength', [])) > 0:
                    l, = plt.loglog(sed['wavelength'], sed['flux'], '--', label=f"{name} (Obs)", alpha=0.7)
                    color = l.get_color()
                    
                    # 2. Absolute SED (10pc) -> Continuous
                    # Calculate if not present
                    if dist is not None:
                        # Scaling factor: (D / 10pc)^2
                        # Flux_10pc = Flux_obs * (D / 10pc)^2
                        # Wait. Flux goes as 1/r^2. 
                        # L = 4*pi*d^2 * F_obs. 
                        # F_10pc = L / (4*pi*10pc^2) = F_obs * (d / 10pc)^2. Correct.
                        factor = ((dist / (10.0 * u.pc))**2).decompose()
                        flux_10pc = sed['flux'] * factor
                        
                        current_max = np.max(flux_10pc)
                        if hasattr(current_max, 'value'): current_max = current_max.value
                        if current_max > global_max_flux: global_max_flux = current_max
                        
                        plt.loglog(sed['wavelength'], flux_10pc, '-', color=color, label=f"{name} (Abs @10pc)")
                        
                # 3. Photometry -> Markers
                if len(photo.get('wavelength', [])) > 0:
                    col = color if color else 'red'
                    plt.errorbar(photo['wavelength'], photo['flux'], 
                                 yerr=photo.get('flux_error'), 
                                 fmt='o', color=col, ecolor=col, markersize=5, capsize=3, label='_nolegend_')

            elif obj_type == "SolarSystem":
                data = get_solar_system_properties(name, force=False)
                if not data: continue
                
                # We need to perform Hybrid stitching (Real + Synthetic Thermal)
                # 1. Get Real Data (if any)
                sed_data = data.get('sed', {})
                wl_real = []
                flux_real = []
                if sed_data and len(sed_data.get('wavelength', [])) > 0:
                     wl_real = sed_data['wavelength']
                     flux_real = sed_data['flux']
                
                # Need params: parameters are in data['physics'] usually or CONSTANTS
                from helios.io.external_query.solar_system.constants import SOLAR_SYSTEM_DATA
                
                physics = data.get('physics', {})
                radius = physics.get('radius')
                teff = physics.get('temperature_eff')
                albedo = physics.get('albedo') 
                
                # Distance: Planet to Sun (semi-major axis) and Planet to Obs (distance)
                # 'r' in ephemeris is distance to Sun? 'delta' is distance to Earth.
                # data['ephemeris']['coordinates']['r'] -> Sun-Planet dist
                # data['ephemeris']['coordinates']['delta'] -> Obs-Planet dist
                
                ephem = data.get('ephemeris', {}).get('coordinates', {})
                dist_sun = ephem.get('r')
                dist_obs = ephem.get('delta')
                
                # FALLBACK to Constants if missing (e.g., failed JPL query)
                if name in SOLAR_SYSTEM_DATA:
                    defaults = SOLAR_SYSTEM_DATA[name]
                    if radius is None: radius = defaults.get('radius')
                    if teff is None: teff = defaults.get('teff')
                    if albedo is None: albedo = defaults.get('albedo')
                    
                    # Approximations for distance if missing
                    # Use Semi-Major Axis for dist_sun if r is missing?
                    # Or just 1 AU if Earth, 5.2 AU if Jupiter etc.
                    # We don't have 'a' in CONSTANTS? We can infer or add.
                    # Let's assume r ~ a. S_p = S_sun * (1/r_au)^2.
                    if dist_sun is None:
                        # Rude approximation map
                        avg_dists = {'Earth': 1.0*u.AU, 'Jupiter': 5.2*u.AU, 'Mercury': 0.4*u.AU, 'Venus': 0.7*u.AU, 'Mars': 1.5*u.AU}
                        dist_sun = avg_dists.get(name, 1.0*u.AU)
                
                # If we want Absolute @ 10pc, we override dist_obs
                dist_metric = 10.0 * u.pc
                
                # If we lack parameters, skip synth but plot Real if avail
                if radius is not None and teff is not None and dist_sun is not None:
                     from helios.sim.spectrum import simulate_lit_planet
                     from helios.io.external_query.solar_system.spectrum import get_solar_spectrum
                     
                     # Common Grid for Hybrid
                     wl_grid = np.logspace(np.log10(0.1), np.log10(500.0), 1000) * u.um
                     
                     # Get Sun
                     sun_spec = get_solar_spectrum() # (wl, flux@1AU)
                     
                     if albedo is None: albedo = 0.3 # Default
                     
                     # Simulate
                     # We use dist_metric (10pc) for the output flux level
                     flux_total, flux_refl, flux_therm = simulate_lit_planet(
                         wl_grid, sun_spec, dist_sun, radius, dist_metric, float(albedo), teff
                     )
                     
                     # 3. Stitch
                     
                     if len(wl_real) > 0:
                         # Interpolate Real to Grid
                         flux_real_interp = np.interp(wl_grid.value, wl_real.to(u.um).value, flux_real.to(u.Jy).value, left=0, right=0) * u.Jy
                         
                         # Mask where Real Data is valid
                         min_real, max_real = np.min(wl_real), np.max(wl_real)
                         in_range = (wl_grid >= min_real) & (wl_grid <= max_real)
                         
                         # Construct Hybrid
                         flux_hybrid = flux_therm.copy() # Start with Thermal
                         
                         flux_hybrid[in_range] += flux_real_interp[in_range]
                         # Outside range: Synthetic Reflected tail + Thermal
                         flux_hybrid[~in_range] += flux_refl[~in_range] 
                         
                         final_flux = flux_hybrid
                         label_txt = f"{name} (Hybrid: Real+Synth)"
                     else:
                         # Fully Synthetic
                         final_flux = flux_total
                         label_txt = f"{name} (Synthetic)"
                         
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
                
                # 1. Absolute SED (10pc) -> Continuous
                # Cache has no SED. We generate it at runtime.
                # Need physics
                if 'physics' in data:
                    # We need a wavelength grid. Let's use 0.1 to 30 um
                    wl_grid = np.logspace(np.log10(0.1), np.log10(30.0), 500) * u.um
                    
                    # We need to pass the physics dict to the generator
                    # But wait, generate_exoplanet_spectrum might calculate Total Flux (Reflected + Thermal).
                    # We need to check if it returns Observed or Surface flux.
                    # Usually it returns Spectral Flux Density.
                    
                    # Let's inspect generate_exoplanet_spectrum signature or re-implement simple BB here.
                    # Re-implementing safer to control distance.
                    
                    # Thermal Emission (Blackbody)
                    T_eq = data['physics'].get('temperature_eq')
                    R_pl = data['physics'].get('radius')
                    
                    if T_eq is not None and R_pl is not None:
                        from astropy.modeling.models import BlackBody
                        bb = BlackBody(temperature=T_eq)
                        flux_surface = bb(wl_grid).to(u.W / (u.m**2 * u.um * u.sr), equivalencies=u.spectral_density(wl_grid))
                        
                        # Abs Flux at 10pc
                        dist_10pc = 10.0 * u.pc
                        solid_angle = (np.pi * (R_pl / dist_10pc)**2).decompose() * u.sr
                        flux_10pc = (flux_surface * solid_angle).to(u.Jy, equivalencies=u.spectral_density(wl_grid))
                        
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
    plt.ylabel('Flux Density (Jy)')
    plt.title('Absolute Spectral Energy Distributions (Normalized to 10 pc)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # Set limits
    if global_max_flux > 0:
        plt.ylim(bottom=1e-10, top=global_max_flux * 1.1)
    else:
        plt.ylim(bottom=1e-10)
    
    # Set X-limits (Restored to 20um as requested)
    plt.xlim(0.1, 20.0)
    
    plt.tight_layout()
    
    output_file = "sed_demo_plot.png"
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
