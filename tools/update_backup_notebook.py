import json

# Read the notebook
with open('../demo_backup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the coronagraph cell by searching for distinctive content
target_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'coro_vortex = helios.Coronagraph' in source and 'Coronagraphic Imaging' not in source:
            target_cell_idx = i
            print(f"Trouvé cellule coronagraphique à l'index {i}")
            break

if target_cell_idx is None:
    print("❌ Cellule coronagraphique non trouvée")
    exit(1)

# New corrected code
new_code = '''import sys
sys.path.insert(0, './src')
import helios
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

scene = helios.Scene(distance=10*u.pc)
star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
planet = helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
scene.add(star)
scene.add(planet)

print(f"Star flux at 550nm: {star.flux_at(550e-9 * u.m)}")
print(f"Planet flux at 550nm: {planet.flux_at(550e-9 * u.m)}")
print(f"Planet/Star contrast at 550nm: {planet.flux_at(550e-9 * u.m)/star.flux_at(550e-9 * u.m):.1e}")
scene.plot()
plt.show()

# Physical parameters
lam = 550e-9 * u.m
D = 6.5 * u.m
fov = 0.3 * u.arcsec  # Larger FOV to see planet at 0.1 arcsec

# Create coronagraph instances
coro_vortex = helios.Coronagraph(phase_mask='vortex')
coro_4q = helios.Coronagraph(phase_mask='4quadrants')

# Render the scene to an image
scene_img, x, y = scene.render(npix=256, fov=fov, return_coords=True)
extent = [x[0].value, x[-1].value, y[0].value, y[-1].value]

# Apply coronagraphs WITHOUT normalization to preserve flux distribution
img_vortex = coro_vortex.image_from_scene(scene_img, soft=True, oversample=4, 
                                          normalize=False, lam=lam, diameter=D, fov=fov)
img_4q = coro_4q.image_from_scene(scene_img, soft=True, oversample=4, 
                                  normalize=False, lam=lam, diameter=D, fov=fov)

# Find planet position in pixels (1 AU at 10 pc = 0.1 arcsec)
planet_angle = 0.1  # arcsec
pixel_scale = fov.to(u.arcsec).value / 256
planet_idx_x = int(256/2 + planet_angle / pixel_scale)
planet_idx_y = 256//2

print(f"\\nPlanet position: ({planet_idx_x}, {planet_idx_y}) pixels")
print(f"Pixel scale: {pixel_scale:.4f} arcsec/pixel")

# Boost planet ONLY in coronagraph images for visualization
boost_factor = 1e5
radius_px = 5
sigma = 2.0

# Create boosted versions for coronagraph panels
img_vortex_boosted = img_vortex.copy()
img_4q_boosted = img_4q.copy()

for dy in range(-radius_px, radius_px+1):
    for dx in range(-radius_px, radius_px+1):
        yy = planet_idx_y + dy
        xx = planet_idx_x + dx
        if 0 <= yy < 256 and 0 <= xx < 256:
            r2 = dx**2 + dy**2
            if r2 <= radius_px**2:
                gaussian_weight = np.exp(-r2 / (2*sigma**2))
                # Boost only coronagraph images
                img_vortex_boosted[yy, xx] += boost_factor * gaussian_weight * scene_img[planet_idx_y, planet_idx_x]
                img_4q_boosted[yy, xx] += boost_factor * gaussian_weight * scene_img[planet_idx_y, planet_idx_x]

print(f"✓ Planète boostée artificiellement d'un facteur {boost_factor:.1e} pour visualisation coronographique")

# Display comparison with logarithmic scale
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original scene (linear scale, no boost)
vmin_scene = 0
vmax_scene = scene_img.max()
im0 = axes[0].imshow(scene_img, origin='lower', cmap='gray', extent=extent,
                     vmin=vmin_scene, vmax=vmax_scene)
axes[0].set_title('Original Scene (linear scale)')
axes[0].set_xlabel('arcsec')
axes[0].set_ylabel('arcsec')
axes[0].plot([planet_angle], [0], 'rx', markersize=10, label='Planet (1 AU)')
plt.colorbar(im0, ax=axes[0], label='Relative Intensity')
axes[0].legend()

# Vortex coronagraph (log scale, with boost)
img_vortex_display = np.log10(img_vortex_boosted + 1e-10)
im1 = axes[1].imshow(img_vortex_display, origin='lower', cmap='inferno', extent=extent)
axes[1].set_title('Vortex Coronagraph (log scale)')
axes[1].set_xlabel('arcsec')
axes[1].set_ylabel('arcsec')
axes[1].plot([planet_angle], [0], 'rx', markersize=10, label='Planet (boosted)')
plt.colorbar(im1, ax=axes[1], label='log10(Intensity)')
axes[1].legend()

# 4-Quadrant coronagraph (log scale, with boost)
img_4q_display = np.log10(img_4q_boosted + 1e-10)
im2 = axes[2].imshow(img_4q_display, origin='lower', cmap='inferno', extent=extent)
axes[2].set_title('4-Quadrant Coronagraph (log scale)')
axes[2].set_xlabel('arcsec')
axes[2].set_ylabel('arcsec')
axes[2].plot([planet_angle], [0], 'rx', markersize=10, label='Planet (boosted)')
plt.colorbar(im2, ax=axes[2], label='log10(Intensity)')
axes[2].legend()

for ax in axes:
    ax.axvline(0, color='white', ls='--', alpha=0.3, lw=0.5)
    ax.axhline(0, color='white', ls='--', alpha=0.3, lw=0.5)

plt.tight_layout()
plt.show()

# Diagnostics
print(f"\\nScene dynamic range: {scene_img.max()/scene_img.min():.1e}")
print(f"Vortex suppression at center: {scene_img.max()/img_vortex[128,128]:.1e}x")
print(f"Peak intensity at planet location (vortex): {img_vortex[planet_idx_y, planet_idx_x]:.2e}")
'''

# Split into lines for notebook format
nb['cells'][target_cell_idx]['source'] = [line + '\n' for line in new_code.split('\n')[:-1]] + [new_code.split('\n')[-1]]

# Clear outputs
nb['cells'][target_cell_idx]['outputs'] = []
nb['cells'][target_cell_idx]['execution_count'] = None

# Save the updated notebook
with open('../demo_backup_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook mis à jour: demo_backup_fixed.ipynb")
