"""
08_end_to_end_simulation.py

Runs a full end-to-end simulation: Scene -> Collectors -> Camera.
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
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
    planet = helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
    scene.add(star)
    scene.add(planet)

    # 2. Collectors
    collectors = helios.TelescopeArray(latitude=0*u.deg, longitude=0*u.deg, altitude=2400*u.m)
    pupil_obs = helios.Pupil(8*u.m)
    collectors.add_collector(pupil=pupil_obs, position=(0, 0), size=8*u.m)

    # 3. Camera
    camera = helios.Camera(pixels=(256, 256))

    # 4. Context & Simulation
    context = helios.Context()
    context.add_layer(scene)
    context.add_layer(collectors)
    context.add_layer(camera)

    print("Running simulation...")
    result = context.observe()
    
    print(f"Simulation complete!")
    print(f"Result shape: {result.shape}")
    print(f"Result range: [{result.min():.2e}, {result.max():.2e}]")

    plt.imshow(result, origin='lower', cmap='inferno')
    plt.colorbar(label='Intensity')
    plt.title('Simulated Observation Result')
    plt.show()

if __name__ == "__main__":
    run_demo()
