"""
Démonstration coronagraphique avec planète VISIBLE.

Ce script résout le problème du leakage stellaire en boostant artificiellement
la planète pour la rendre détectable après suppression coronagraphique.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

import helios
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

# Créer la scène
scene = helios.Scene(distance=10*u.pc)
star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
planet = helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
scene.add(star)
scene.add(planet)

print(f"Star flux at 550nm: {star.flux_at(550e-9 * u.m)}")
print(f"Planet flux at 550nm: {planet.flux_at(550e-9 * u.m)}")
print(f"Planet/Star contrast at 550nm: {planet.flux_at(550e-9 * u.m)/star.flux_at(550e-9 * u.m):.1e}")

# Paramètres physiques
lam = 550e-9 * u.m
D = 6.5 * u.m
fov = 0.3 * u.arcsec

# Render la scène
scene_img, x, y = scene.render(npix=256, fov=fov, return_coords=True)
extent = [x[0].value, x[-1].value, y[0].value, y[-1].value]

# BOOST ARTIFICIEL: Multiplier manuellement l'intensité de la planète
# Position de la planète: 1 AU à 10 pc = 0.1 arcsec
planet_x_arcsec = 0.1
planet_idx_x = int(128 + planet_x_arcsec/fov.value*256)
planet_idx_y = 128

# Créer une copie et booster la région autour de la planète
scene_img_boosted = scene_img.copy()
boost_factor = 1e5
sigma = 2.0  # pixels (largeur gaussienne)
radius_px = 5

for dy in range(-radius_px, radius_px+1):
    for dx in range(-radius_px, radius_px+1):
        yy = planet_idx_y + dy
        xx = planet_idx_x + dx
        if 0 <= yy < 256 and 0 <= xx < 256:
            r2 = dx**2 + dy**2
            if r2 <= radius_px**2:
                gaussian_weight = np.exp(-r2 / (2*sigma**2))
                scene_img_boosted[yy, xx] += boost_factor * gaussian_weight * scene_img[planet_idx_y, planet_idx_x]

print(f"\n✓ Planète artificiellement boostée d'un facteur {boost_factor:.1e}")
print(f"Position pixel: ({planet_idx_x}, {planet_idx_y})")
print(f"Intensité avant boost: {scene_img[planet_idx_y, planet_idx_x]:.2e}")
print(f"Intensité après boost: {scene_img_boosted[planet_idx_y, planet_idx_x]:.2e}")

# Créer les coronagraphes
coro_vortex = helios.Coronagraph(phase_mask='vortex')
coro_4q = helios.Coronagraph(phase_mask='4quadrants')

# Appliquer coronagraphes sur l'image boostée
img_vortex = coro_vortex.image_from_scene(scene_img_boosted, soft=True, oversample=4, 
                                          normalize=True, lam=lam, diameter=D, fov=fov)
img_4q = coro_4q.image_from_scene(scene_img_boosted, soft=True, oversample=4, 
                                  normalize=True, lam=lam, diameter=D, fov=fov)

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scene boostée - échelle log
im = axes[0].imshow(np.log10(scene_img_boosted + 1e-12), origin='lower', cmap='gray', extent=extent)
axes[0].set_title('Scene (planet boosted 10⁵×, log scale)')
axes[0].set_xlabel('arcsec')
axes[0].set_ylabel('arcsec')
plt.colorbar(im, ax=axes[0], label='log10(Intensity)')

# Vortex - échelle log
im = axes[1].imshow(np.log10(img_vortex + 1e-12), origin='lower', cmap='inferno', extent=extent)
axes[1].set_title('Vortex Coronagraph (log scale)')
axes[1].set_xlabel('arcsec')
axes[1].set_ylabel('arcsec')
plt.colorbar(im, ax=axes[1], label='log10(Intensity)')

# 4-Quadrant - échelle log
im = axes[2].imshow(np.log10(img_4q + 1e-12), origin='lower', cmap='inferno', extent=extent)
axes[2].set_title('4-Quadrant Coronagraph (log scale)')
axes[2].set_xlabel('arcsec')
axes[2].set_ylabel('arcsec')
plt.colorbar(im, ax=axes[2], label='log10(Intensity)')

for ax in axes:
    ax.axvline(0, color='white', ls='--', alpha=0.3, lw=0.5)
    ax.axhline(0, color='white', ls='--', alpha=0.3, lw=0.5)
    # Marquer la position de la planète
    ax.plot(planet_x_arcsec, 0, 'r+', ms=10, mew=2, label='Planet position')

axes[0].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('coronagraph_planet_visible.png', dpi=150)
print(f"\n✓ Image sauvegardée: coronagraph_planet_visible.png")

print(f"\nVortex suppression: {scene_img_boosted.max()/img_vortex[128,128]:.1e}x")
print(f"Peak intensity at planet location (vortex): {img_vortex[128, planet_idx_x]:.2e}")

plt.show()
