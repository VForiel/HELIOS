"""
09_end_to_end_simulation.py

Runs a full end-to-end simulation with multi-sample Wavefronts:
Scene -> TelescopeArray (collectors) -> Camera. Compatible with 3D
Wavefront/WavefrontArray and updated Context.observe() behavior.
"""
import sys
import os
import matplotlib.pyplot as plt
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios

def run_demo():
    # 1. Scene
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.arcsec, 0*u.arcsec))
    # Place planet off-axis in angular coordinates for interferometric demo
    planet = helios.Planet(mass=1*u.M_jup, position=(0.1*u.arcsec, 0*u.arcsec))
    scene.add(star)
    scene.add(planet)

    # 2. Collectors (simple single-aperture or array)
    collectors = helios.TelescopeArray(name="Simple Array")
    pupil_obs = helios.Pupil(diameter=2*u.m)
    pupil_obs.add_disk(center=(0,0), radius=1*u.m)
    collectors.add_collector(pupil=pupil_obs, position=(0*u.m, 0*u.m), size=2*u.m)

    # 3. Camera
    camera = helios.Camera(pixels=(256, 256))

    # 4. Context & Simulation
    # Specify simulation parameters (wavelength and grid size)
    context = helios.Context(wavelength=600*u.nm, npix=512)
    context.add_layer(scene)
    context.add_layer(collectors)
    context.add_layer(camera)

    print("Running simulation...")
    # observe() internally builds input wavefront; returns camera image (numpy array)
    result = context.observe()
    
    print(f"Simulation complete!")
    print(f"Result shape: {result.shape}")
    print(f"Result range: [{result.min():.2e}, {result.max():.2e}]")

    plt.figure(figsize=(8, 6))
    plt.imshow(result, origin='lower', cmap='inferno')
    plt.colorbar(label='Intensity')
    plt.title('Simulated Observation Result')
    
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
