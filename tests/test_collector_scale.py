import sys
import os
import numpy as np
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

import helios

def test_collector_pixel_scale():
    print("Testing Collector pixel scale update...")
    
    # Create a wavefront with arbitrary initial scale
    wf = helios.Wavefront(wavelength=1e-6*u.m, size=100)
    wf.pixel_scale = 1.0 * u.m # Initial scale (100m total)
    
    # Create a collector with specific size
    D = 8.0 * u.m
    pupil = helios.Pupil(diameter=D)
    pupil.add_disk(radius=D/2)
    collector = helios.Collector(pupil=pupil, position=(0,0), size=D)
    
    # Process wavefront
    wf_processed = collector.process(wf, None)
    
    # Check new pixel scale
    expected_scale = (D.to(u.m).value / 100) * u.m
    print(f"Initial scale: {1.0*u.m}")
    print(f"Collector size: {D}")
    print(f"Processed scale: {wf_processed.pixel_scale}")
    print(f"Expected scale: {expected_scale}")
    
    assert np.isclose(wf_processed.pixel_scale.to(u.m).value, expected_scale.to(u.m).value), \
        f"Pixel scale mismatch! Got {wf_processed.pixel_scale}, expected {expected_scale}"
        
    print("Collector pixel scale test passed.")

if __name__ == "__main__":
    test_collector_pixel_scale()
