
from helios.components.scene import Scene, Star, Planet, Zodiacal
import astropy.units as u
import numpy as np

def check_flux():
    scene = Scene()
    star = Star(temperature=5778*u.K, magnitude=4.83) # Sun at 10pc (roughly)
    scene.add(star)
    
    planet = Planet(mass=1*u.M_jup, orbit_radius=1*u.AU) # Jupiter at 1AU
    scene.add(planet)
    
    zodi = Zodiacal(brightness=1.0)
    scene.add(zodi)
    
    # Grid
    wavelengths = np.logspace(np.log10(0.1), np.log10(100), 200) * u.um
    
    print("--- SED Check ---")
    
    # 1. Star
    wl, sed_star = star.sed(wavelengths=wavelengths)
    # Convert to plot units: W / (m2 um sr)
    sed_star_plot = sed_star.to(u.W / (u.m**2 * u.um * u.sr)).value
    print(f"Star Max: {np.max(sed_star_plot):.2e}")
    
    # 2. Planet
    wl, sed_planet = planet.sed(wavelengths=wavelengths)
    sed_planet_plot = sed_planet.to(u.W / (u.m**2 * u.um * u.sr)).value
    print(f"Planet Max: {np.max(sed_planet_plot):.2e}")
    
    # Ratio
    if np.max(sed_planet_plot) > 0:
        ratio = np.max(sed_star_plot) / np.max(sed_planet_plot)
        print(f"Star/Planet Ratio: {ratio:.2e}")
    else:
        print("Planet Max is 0 or negative!")
        
    # 3. Zodiacal
    wl, sed_zodi = zodi.sed(wavelengths=wavelengths)
    sed_zodi_plot = sed_zodi.to(u.W / (u.m**2 * u.um * u.sr)).value
    print(f"Zodiacal Max: {np.max(sed_zodi_plot):.2e}")
    
    # Check for NaNs
    if np.any(np.isnan(sed_planet_plot)):
        print("Planet SED contains NaNs!")
    
    print("----------------")

if __name__ == "__main__":
    check_flux()
