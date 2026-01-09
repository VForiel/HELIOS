"""
07_interferometry_arrays.py

Demonstrates how to configure interferometric arrays (VLTI, LIFE).
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

import helios

def run_demo(save=False):
    print("Configuring VLTI...")
    vlti = helios.TelescopeArray.vlti(uts=True)

    print(f"Interferometer created: {vlti.name}")
    print(f"Number of collectors: {vlti.num_telescopes}")
    # print(f"Baselines (m):\n{vlti.get_baseline_array()}")

    vlti.plot_array(show_pupils=True, pupil_scale=0.5)
    
    if save:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated'))
        os.makedirs(output_dir, exist_ok=True)
        filename = "07_interferometry_arrays.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close() # Close to free memory
    else:
        plt.show()

if __name__ == "__main__":
    run_demo()
