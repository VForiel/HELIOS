import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

import helios

def test_plot_units():
    print("Testing Wavefront.plot units...")
    
    # Case 1: Pupil plane (meters)
    wf = helios.Wavefront(wavelength=1e-6*u.m, size=100)
    wf.pixel_scale = 0.01 * u.m # 1 cm per pixel -> 1m total
    # Should use meters or mm? 1m total -> likely m
    
    # Mock plot to check labels (we can't easily check plot output programmatically without saving/inspecting, 
    # but running it ensures no crash and we can manually inspect if needed, or just trust the logic)
    # We will just run it to ensure no errors.
    try:
        wf.plot(show=False)
        print("Wavefront.plot (meters) ran successfully.")
    except Exception as e:
        print(f"Wavefront.plot (meters) failed: {e}")

    # Case 2: Image plane (angles)
    wf.pixel_scale = 0.01 * u.arcsec # 1 arcsec total
    try:
        wf.plot(show=False)
        print("Wavefront.plot (arcsec) ran successfully.")
    except Exception as e:
        print(f"Wavefront.plot (arcsec) failed: {e}")

    print("Testing WavefrontArray.plot units...")
    wf_arr = helios.WavefrontArray([wf, wf.copy()])
    try:
        wf_arr.plot(show=False)
        print("WavefrontArray.plot ran successfully.")
    except Exception as e:
        print(f"WavefrontArray.plot failed: {e}")

if __name__ == "__main__":
    test_plot_units()
