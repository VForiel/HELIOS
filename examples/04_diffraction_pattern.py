"""
04_diffraction_pattern.py

Demonstrates how to compute and visualize the diffraction pattern (PSF) of a pupil.
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
    # Load JWST Preset
    print("Loading JWST preset...")
    p_jwst = helios.Pupil.like('JWST')
    arr_jwst = p_jwst.get_array(npix=512)

    # Diffraction Pattern (PSF)
    print("Computing diffraction pattern...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pupil amplitude
    axes[0].imshow(arr_jwst, origin='lower', cmap='gray', extent=[-1, 1, -1, 1])
    axes[0].set_title('JWST Pupil')
    axes[0].set_xlabel('Normalized pupil coordinate')
    axes[0].set_ylabel('Normalized pupil coordinate')

    # Diffraction pattern (PSF)
    axes[1] = p_jwst.plot_diffraction_pattern(npix=512, log=True, cmap='inferno', 
                                              wavelength=550e-9, ax=axes[1])
    axes[1].set_title('JWST Diffraction Pattern (log scale)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()
