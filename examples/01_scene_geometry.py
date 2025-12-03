"""
01_scene_geometry.py

Demonstrates how to define a scene with astronomical objects and visualize their
spatial distribution.
"""
import sys
import os
import matplotlib.pyplot as plt
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios

def run_demo():
    # Create scene
    scene = helios.Scene(distance=10*u.pc)
    
    # Add objects
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
    planet = helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
    zodi = helios.Zodiacal(brightness=0.5)
    exozodi = helios.ExoZodiacal(brightness=0.3)
    
    scene.add(star)
    scene.add(planet)
    scene.add(zodi)
    scene.add(exozodi)

    # Visualize scene geometry
    print("Plotting scene geometry...")
    scene.plot()
    plt.show()

if __name__ == "__main__":
    run_demo()
