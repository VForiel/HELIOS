
from helios.components.scene import Planet, Star, Scene
import astropy.units as u
import numpy as np
import traceback

def debug_planet():
    try:
        print("Creating Scene...")
        scene = Scene()
        star = Star(temperature=5000*u.K, magnitude=5)
        scene.add(star)
        
        print("Creating Planet...")
        planet = Planet(mass=1*u.M_jup, orbit_radius=1*u.AU)
        scene.add(planet)
        
        print("Calculating Planet SED...")
        # wl, sed = planet.sed() # This fails, so let's break it down
        
        # 1. Thermal
        print("1. Thermal Component")
        # Planet inherits from SceneObject which uses modified_blackbody
        # We can call planet.sed(include_reflection=False) 
        wl_thermal, sed_thermal = planet.sed(include_reflection=False)
        print(f"   Thermal: Shape={sed_thermal.shape}, Unit={sed_thermal.unit}")
        print(f"   Thermal Min/Max: {sed_thermal.min().value:.2e} / {sed_thermal.max().value:.2e}")
        
        # 2. Reflection
        print("2. Reflection Component")
        stars = [obj for obj in scene.objects if isinstance(obj, Star)]
        star = stars[0]
        wl_star, sed_star = star.sed(wavelengths=wl_thermal)
        print(f"   Star SED: Unit={sed_star.unit}")
        
        # Calc scale
        print("   Calculating scale...")
        # From code: reflection_scale = self.albedo * (self.radius / separation)**2
        # Use simple mode first
        separation = 1.0 * u.AU # simplified
        print(f"   Planet Radius: {planet.radius}, Albedo: {planet.albedo}")
        
        # Need to handle unitless/quantity mix
        # In scene.py code:
        # px, py = self.position ...
        # separation = ...
        # reflection_scale = self.albedo * (self.radius / separation)**2
        # reflection_scale = reflection_scale.decompose().value
        
        scale_q = planet.albedo * (planet.radius / separation)**2
        scale = scale_q.decompose().value
        print(f"   Scale: {scale} (Type: {type(scale)})")
        
        sed_reflected = sed_star * scale
        print(f"   Reflected SED: Unit={sed_reflected.unit}")
        
        # 3. Sum
        print("3. Summation")
        sed_total = sed_thermal + sed_reflected
        print(f"   Total SED: Unit={sed_total.unit}")
        print("SUCCESS Planet")

        print("-" * 20)
        print("Testing Zodiacal...")
        from helios.components.scene import Zodiacal
        zodi = Zodiacal(brightness=1.0)
        wl_z, sed_z = zodi.sed()
        print(f"   Zodiacal SED: Unit={sed_z.unit}")
        print(f"   Min/Max: {sed_z.min().value} / {sed_z.max().value}")
        print("SUCCESS Zodiacal")
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    debug_planet()
