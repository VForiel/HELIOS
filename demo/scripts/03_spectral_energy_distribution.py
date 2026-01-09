"""
02_spectral_energy_distribution.py

Demonstrates how to visualize the Spectral Energy Distributions (SEDs) of
astronomical objects.
"""
import sys
import os
import matplotlib.pyplot as plt
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import helios

def run_demo():
    # Create objects
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
    planet = helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU), albedo=0.3, radius=1*u.R_jup)
    
    # Add to scene to enable reflection calculation
    scene.add(star)
    scene.add(planet)

    # Plot SEDs
    print("Plotting Spectral Energy Distributions...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = star.plot_sed(ax=ax, color='gold', label='Star')
    ax = planet.plot_sed(ax=ax, color='blue', label='Planet')
    ax.set_title('Spectral Energy Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
