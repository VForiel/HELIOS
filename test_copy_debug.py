
import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import numpy as np
from astropy import units as u
from helios.core.simulation import Wavefront

def test_copy():
    wf = Wavefront(wavelength=1*u.um, npix=32)
    wf.source_directions = np.array([[1e-6, 0.0]]) * u.rad
    
    wf_copy = wf.copy()
    
    print(f"Original source_directions: {wf.source_directions}")
    print(f"Copy source_directions: {wf_copy.source_directions}")
    
    if wf_copy.source_directions is None:
        print("FAIL: source_directions lost in copy")
    else:
        print("PASS: source_directions preserved")

def test_view():
    wf = Wavefront(wavelength=1*u.um, npix=32)
    wf.source_directions = np.array([[1e-6, 0.0]]) * u.rad
    
    wf_view = wf[np.newaxis, ...]
    
    print(f"View source_directions: {wf_view.source_directions}")
    
    if wf_view.source_directions is None:
        print("FAIL: source_directions lost in view")
    else:
        print("PASS: source_directions preserved in view")

if __name__ == "__main__":
    test_copy()
    test_view()
