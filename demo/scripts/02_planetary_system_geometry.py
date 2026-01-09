"""
02_scene_geometry.py

Demonstrates how to define a planetary system configuration (PlanetarySystem)
and visualize the spatial distribution of its components.
"""
import sys
import os
import matplotlib.pyplot as plt
from astropy import units as u

import helios

def run_demo(save=False):
    # Create PlanetarySystem (replaces old Scene)
    system = helios.PlanetarySystem(distance=10*u.pc, name="Demo System")
    
    # Add Central Star
    star = helios.Star(mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
    system.add(star)

    # Add Planets
    # Planet b: Earth-like at 1 AU
    planet_b = helios.Planet(mass=1*u.M_earth, position=(1*u.AU, 0*u.AU), name="Earth-like")
    # Planet c: Jupiter-like at 5 AU, rotated 90 degrees
    planet_c = helios.Planet(mass=1*u.M_jup, position=(0*u.AU, 5*u.AU), name="Jupiter-like")
    
    system.add(planet_b)
    system.add(planet_c)
    
    # Add Dust Components
    system.add(helios.Zodiacal(brightness=0.5))
    system.add(helios.ExoZodiacal(brightness=0.3))

    # Visualize geometry
    print(f"Plotting system: {system.name}")
    fig, ax = system.plot()
    
    if save:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated'))
        os.makedirs(output_dir, exist_ok=True)
        filename = "02_scene_geometry.png"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    run_demo()
