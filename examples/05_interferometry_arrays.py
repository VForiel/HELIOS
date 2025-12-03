"""
03_interferometry.py

Demonstrates how to configure interferometric arrays (VLTI, LIFE) and visualize
their baselines.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios

def run_demo():
    # 1. VLTI Configuration
    print("Configuring VLTI...")
    vlti = helios.TelescopeArray.vlti(uts=True)

    print(f"Interferometer created: {vlti.name}")
    print(f"Number of collectors: {len(vlti.collectors)}")
    print(f"Baselines (m):\n{vlti.get_baseline_array()}")

    vlti.plot_array(show_pupils=True, pupil_scale=0.5)
    plt.show()

    # 2. LIFE Space Interferometer
    print("Configuring LIFE Space Interferometer...")
    life = helios.TelescopeArray.life()

    print(f"Interferometer created: {life.name}")
    print(f"Location (space): {life.latitude}, {life.longitude}")
    print(f"Number of collectors: {len(life.collectors)}")
    print(f"Baselines (m):\n{life.get_baseline_array()}")

    # Calculate maximum baseline
    baselines = life.get_baseline_array()
    max_bl = max([np.linalg.norm(baselines[i] - baselines[j]) 
                  for i in range(len(baselines)) 
                  for j in range(i+1, len(baselines))])
    print(f"Maximum baseline: {max_bl:.2f} m")

    life.plot_array(show_pupils=True, pupil_scale=0.1)
    plt.show()

if __name__ == "__main__":
    run_demo()
