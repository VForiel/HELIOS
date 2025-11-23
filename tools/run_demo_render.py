# Headless demo renderer for CI-style checks
import sys
import os
# ensure local src is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy import units as u
import helios
from helios.components.scene import Scene
from helios.components.optics import Pupil


def main():
    scene = helios.components.Scene(distance=10*u.pc)
    star = helios.components.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
    planet = helios.components.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
    scene.add(star)
    scene.add(planet)
    zodi = helios.components.Zodiacal(brightness=0.5)
    exozodi = helios.components.ExoZodiacal(brightness=0.3)
    scene.add(zodi)
    scene.add(exozodi)

    print('Rendering scene...')
    img, x, y = scene.render(npix=256, fov=4*u.arcsec, return_coords=True)
    print('Scene render shape:', img.shape, 'x[0],x[-1]:', x[0], x[-1])

    out1 = os.path.join('examples', 'demo_scene_render.png')
    plt.imsave(out1, img, cmap='gray', origin='lower')
    print('Saved', out1)

    # propagate through JWST pupil
    p_jwst = Pupil.like('JWST')
    print('Computing image through pupil...')
    img2 = p_jwst.image_through_pupil(img, soft=True)
    out2 = os.path.join('examples', 'demo_scene_image_through_pupil.png')
    plt.imsave(out2, img2, cmap='inferno', origin='lower')
    print('Saved', out2)


if __name__ == '__main__':
    main()
