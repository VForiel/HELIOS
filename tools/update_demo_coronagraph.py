import json

# Read the notebook
with open('demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 27 (the second coronagraph cell) by searching for distinctive content
target_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Planet(mass=10*u.M_jup, position=(4*u.AU' in source and 'coro_vortex = helios.Coronagraph' in source:
            target_cell_idx = i
            print(f"✓ Trouvé cellule coronagraphique (10 M_jup) à l'index {i}")
            break

if target_cell_idx is None:
    print("❌ Cellule non trouvée")
    exit(1)

# New code with all improvements
new_code = '''import sys
sys.path.insert(0, './src')
import helios
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

scene = helios.Scene(distance=10*u.pc)
star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
planet = helios.Planet(mass=10*u.M_jup, position=(4*u.AU, 0*u.AU))
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
fov = 1 * u.arcsec

# Create coronagraph instances
coro_vortex = helios.Coronagraph(phase_mask='vortex')
coro_4q = helios.Coronagraph(phase_mask='4quadrants')

# Render the scene to an image
scene_img, x, y = scene.render(npix=256, fov=fov, return_coords=True)
extent = [x[0].value, x[-1].value, y[0].value, y[-1].value]

# Apply coronagraphs WITHOUT normalization to preserve physical flux distribution
img_vortex = coro_vortex.image_from_scene(scene_img, soft=True, oversample=4, 
                                          normalize=False, lam=lam, diameter=D, fov=fov)
img_4q = coro_4q.image_from_scene(scene_img, soft=True, oversample=4, 
                                  normalize=False, lam=lam, diameter=D, fov=fov)

# Calculate suppression ratios
max_before = scene_img.max()
max_after_vortex = img_vortex.max()
max_after_4q = img_4q.max()
suppression_vortex = max_before / max_after_vortex
suppression_4q = max_before / max_after_4q

print(f"\\n--- Coronagraph Performance ---")
print(f"Vortex suppression: {suppression_vortex:.1e}x")
print(f"4-Quadrant suppression: {suppression_4q:.1e}x")

# Find planet position (4 AU at 10 pc = 0.4 arcsec)
planet_angle = 0.4  # arcsec
pixel_scale = fov.to(u.arcsec).value / 256
planet_idx_x = int(256/2 + planet_angle / pixel_scale)
planet_idx_y = 256//2

print(f"\\n--- Planet Detection ---")
print(f"Planet position: ({planet_idx_x}, {planet_idx_y}) pixels = {planet_angle} arcsec")
print(f"Original scene at planet: {scene_img[planet_idx_y, planet_idx_x]:.2e}")
print(f"Vortex image at planet: {img_vortex[planet_idx_y, planet_idx_x]:.2e}")
print(f"4-Quadrant at planet: {img_4q[planet_idx_y, planet_idx_x]:.2e}")
print(f"⚠️ La planète n'est PAS visible après coronagraphe car elle est trop proche du centre")
print(f"   (le coronagraphe supprime TOUT dans un rayon ~2 λ/D, incluant la planète)")

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original scene
im0 = axes[0].imshow(scene_img, origin='lower', cmap='gray', extent=extent)
axes[0].set_title('Original Scene')
axes[0].set_xlabel('arcsec')
axes[0].set_ylabel('arcsec')
axes[0].plot([planet_angle], [0], 'rx', markersize=10, label=f'Planet ({planet.mass.value:.0f} M_jup, 4 AU)')
cbar0 = plt.colorbar(im0, ax=axes[0], label='Intensity', fraction=0.046, pad=0.04)
axes[0].legend(loc='upper right')

# Vortex coronagraph
im1 = axes[1].imshow(img_vortex, origin='lower', cmap='inferno', extent=extent)
axes[1].set_title(f'Vortex Coronagraph\\n(Suppression: {suppression_vortex:.1e}x)')
axes[1].set_xlabel('arcsec')
axes[1].set_ylabel('arcsec')
axes[1].plot([planet_angle], [0], 'cx', markersize=10, label='Planet (suppressed)')
cbar1 = plt.colorbar(im1, ax=axes[1], label='Intensity', fraction=0.046, pad=0.04)
axes[1].legend(loc='upper right')

# 4-Quadrant coronagraph
im2 = axes[2].imshow(img_4q, origin='lower', cmap='inferno', extent=extent)
axes[2].set_title(f'4-Quadrant Coronagraph\\n(Suppression: {suppression_4q:.1e}x)')
axes[2].set_xlabel('arcsec')
axes[2].set_ylabel('arcsec')
axes[2].plot([planet_angle], [0], 'cx', markersize=10, label='Planet (suppressed)')
cbar2 = plt.colorbar(im2, ax=axes[2], label='Intensity', fraction=0.046, pad=0.04)
axes[2].legend(loc='upper right')

for ax in axes:
    ax.axvline(0, color='white', ls='--', alpha=0.3, lw=0.5)
    ax.axhline(0, color='white', ls='--', alpha=0.3, lw=0.5)

plt.tight_layout()
plt.show()

print(f"\\nScene dynamic range: {scene_img.max()/scene_img.min():.1e}")
'''

# Split into lines for notebook format
nb['cells'][target_cell_idx]['source'] = [line + '\n' for line in new_code.split('\n')[:-1]] + [new_code.split('\n')[-1]]

# Clear outputs to force re-execution
nb['cells'][target_cell_idx]['outputs'] = []
nb['cells'][target_cell_idx]['execution_count'] = None

# Save
with open('demo_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook mis à jour: demo_fixed.ipynb")
