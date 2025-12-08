
import astropy.units as u
import numpy as np
from helios.components.scene import Planet, Star, Scene

def test_unit_conversion():
    try:
        # 1. Setup
        wavelengths = np.logspace(np.log10(0.1), np.log10(100), 200) * u.um
        print(f"Wavelengths: {wavelengths.unit}")
        
        target_unit = u.W / (u.m**2 * u.um * u.sr)
        print(f"Target Unit: {target_unit}")
        
        # 2. Test Planet with wavelengths
        planet = Planet(mass=1*u.M_jup, orbit_radius=1*u.AU)
        scene = Scene()
        scene.add(planet)
        star = Star(temperature=5778*u.K, magnitude=4.83)
        scene.add(star)
        
        print("Calling planet.sed(wavelengths)...")
        wl, sed = planet.sed(wavelengths=wavelengths)
        print(f"Returned SED Unit: {sed.unit}")
        
        print("Converting...")
        sed_conv = sed.to(target_unit)
        print("Conversion Successful")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_unit_conversion()
