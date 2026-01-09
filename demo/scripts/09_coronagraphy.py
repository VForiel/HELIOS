"""
06_coronagraphy.py

Demonstrates the effect of coronagraphic phase masks (Vortex, 4-Quadrant) on
starlight suppression and planet detection.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import helios

def run_demo():
    # Setup Scene
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
    planet = helios.Planet(mass=10*u.M_jup, position=(4*u.AU, 0*u.AU))
    scene.add(star)
    scene.add(planet)

    print(f"Planet/Star contrast: {planet.flux_at(550e-9 * u.m)/star.flux_at(550e-9 * u.m):.1e}")

    # Parameters
    lam = 550e-9 * u.m
    D = 6.5 * u.m
    fov = 1 * u.arcsec

    # Coronagraphs
    coro_vortex = helios.Coronagraph(phase_mask='vortex')
    coro_4q = helios.Coronagraph(phase_mask='4quadrants')

    # Render Scene
    print("Rendering scene...")
    scene_img, x, y = scene.render(npix=256, fov=fov, return_coords=True)
    extent = [x[0].value, x[-1].value, y[0].value, y[-1].value]

    # Apply Coronagraphs
    print("Applying coronagraphs...")
    img_vortex = coro_vortex.image_from_scene(scene_img, soft=True, oversample=4, 
                                              normalize=False, lam=lam, diameter=D, fov=fov)
    img_4q = coro_4q.image_from_scene(scene_img, soft=True, oversample=4, 
                                      normalize=False, lam=lam, diameter=D, fov=fov)

    # Calculate suppression
    suppression_vortex = scene_img.max() / img_vortex.max()
    suppression_4q = scene_img.max() / img_4q.max()
    print(f"Vortex suppression: {suppression_vortex:.1e}x")
    print(f"4-Quadrant suppression: {suppression_4q:.1e}x")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original
    im0 = axes[0].imshow(scene_img, origin='lower', cmap='gray', extent=extent)
    axes[0].set_title('Original Scene')
    plt.colorbar(im0, ax=axes[0], label='Intensity')

    # Vortex
    im1 = axes[1].imshow(img_vortex, origin='lower', cmap='inferno', extent=extent)
    axes[1].set_title(f'Vortex Coronagraph\n(Suppression: {suppression_vortex:.1e}x)')
    plt.colorbar(im1, ax=axes[1], label='Intensity')

    # 4Q
    im2 = axes[2].imshow(img_4q, origin='lower', cmap='inferno', extent=extent)
    axes[2].set_title(f'4-Quadrant Coronagraph\n(Suppression: {suppression_4q:.1e}x)')
    plt.colorbar(im2, ax=axes[2], label='Intensity')

    plt.tight_layout()
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated/examples'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    run_demo()
