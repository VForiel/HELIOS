import sys
import numpy as np
from astropy import units as u
sys.path.insert(0, 'src')

import helios
from helios.components import Scene, Star, Planet, Zodiacal, ExoZodiacal

def summarize(name, wl, sed):
    print(f"--- {name} ---")
    print("wavelengths type:", type(wl), "unit:", getattr(wl, 'unit', None))
    print("sed type:", type(sed), "unit:", getattr(sed, 'unit', None))
    try:
        arr_wl = wl.to(u.um).value
        arr_sed = sed.to(u.W / (u.m**2 * u.um * u.sr)).value
        print("len:", len(arr_wl))
        print("wl min/max (um):", arr_wl.min(), arr_wl.max())
        print("sed[0..4]:", arr_sed[:5])
    except Exception as e:
        print("Error summarizing:", e)


def main():
    scene = Scene(distance=10*u.pc)
    star = Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
    planet = Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
    zodi = Zodiacal(brightness=0.5)
    exozodi = ExoZodiacal(brightness=0.3)
    scene.add(star)
    scene.add(planet)
    scene.add(zodi)
    scene.add(exozodi)

    print('Objects in scene:', [type(o).__name__ for o in scene.objects])

    wl_s, sed_s = star.sed()
    summarize('Star', wl_s, sed_s)

    wl_p, sed_p = planet.sed()
    summarize('Planet', wl_p, sed_p)

    wl_z, sed_z = zodi.sed()
    summarize('Zodiacal', wl_z, sed_z)

    wl_x, sed_x = exozodi.sed()
    summarize('ExoZodiacal', wl_x, sed_x)

if __name__ == '__main__':
    main()
