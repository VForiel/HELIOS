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
    
    print("Configuring VLTI...")
    vlti = helios.TelescopeArray.vlti(uts=True)

    print(f"Interferometer created: {vlti.name}")
    print(f"Number of collectors: {len(vlti.collectors)}")
    print(f"Baselines (m):\n{vlti.get_baseline_array()}")

    vlti.plot_array(show_pupils=True, pupil_scale=0.5)
    plt.show()

if __name__ == "__main__":
    run_demo()
