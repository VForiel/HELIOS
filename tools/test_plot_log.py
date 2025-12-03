import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

import helios

def test_plot_log():
    print("Testing Wavefront.plot with log_scale=True...")
    wf = helios.Wavefront(wavelength=1e-6*u.m, size=100)
    wf.pixel_scale = 0.01 * u.m
    
    # Create some structure to see log effect
    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    wf.field = np.exp(-R**2 / 0.1) * np.exp(1j * R * 10)
    
    try:
        wf.plot(title="Test Log Scale", log_scale=True, show=False)
        print("Wavefront.plot(log_scale=True) ran successfully.")
    except Exception as e:
        print(f"Wavefront.plot(log_scale=True) failed: {e}")

    print("Testing WavefrontArray.plot with log_scale=True...")
    wf_arr = helios.WavefrontArray([wf, wf.copy()])
    try:
        wf_arr.plot(title="Test Array Log Scale", log_scale=True, show=False)
        print("WavefrontArray.plot(log_scale=True) ran successfully.")
    except Exception as e:
        print(f"WavefrontArray.plot(log_scale=True) failed: {e}")

if __name__ == "__main__":
    test_plot_log()
