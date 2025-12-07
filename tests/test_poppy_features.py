
import sys
import os
import numpy as np
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

import helios
from helios.core.simulation import PlaneType

def test_poppy_features():
    print("Testing POPPY-inspired features...")
    
    # 1. Create Wavefront
    wf = helios.Wavefront(wavelength=1e-6*u.m, size=1*u.m, npix=256)
    
    print(f"Initial PlaneType: {wf.planetype}")
    assert wf.planetype == PlaneType.PUPIL
    
    print(f"Initial History: {wf.history}")
    assert len(wf.history) > 0
    
    # 2. Mock Lens application (manually setting focal length)
    focal_length = 10.0
    wf._last_focal_length_m = focal_length
    wf.history.append(f"Applied Lens f={focal_length}m")
    
    # 3. Propagate
    print("Propagating...")
    wf_image = wf.propagate(distance=None) # Should use stored focal length
    
    print(f"Final PlaneType: {wf_image.planetype}")
    assert wf_image.planetype == PlaneType.IMAGE
    
    print(f"Final History: {wf_image.history}")
    assert "Propagated to Image Plane" in wf_image.history[-1]
    
    # 4. Check display alias
    print("Checking display() alias...")
    try:
        # Don't actually show plot, just check if method exists and runs without error (mocking plt.show)
        import matplotlib.pyplot as plt
        plt.show = lambda: None
        wf_image.display(show=False)
        print("display() method works.")
    except Exception as e:
        print(f"display() failed: {e}")
        raise
    
    # 5. Test Properties
    print("Testing properties (amplitude, intensity, phase)...")
    amp = wf_image.amplitude
    inte = wf_image.intensity
    phi = wf_image.phase
    
    assert isinstance(amp, u.Quantity)
    assert isinstance(inte, u.Quantity)
    assert isinstance(phi, u.Quantity)
    assert phi.unit.is_equivalent(u.rad)
    
    print(f"Total Intensity: {wf_image.total_intensity}")
    
    # 6. Test Coordinates
    print("Testing coordinates()...")
    y, x = wf_image.coordinates()
    assert y.shape == (256, 256)
    assert x.shape == (256, 256)
    assert y.unit == wf_image.pixel_scale.unit
    print(f"Coordinate range: {x.min()} to {x.max()}")

    # 7. Test Tilt and Rotate
    print("Testing tilt and rotate...")
    wf_tilt = wf.copy()
    wf_tilt.tilt(x_angle=1*u.arcsec)
    assert "Tilted by" in wf_tilt.history[-1]
    
    wf_rot = wf.copy()
    wf_rot.rotate(45*u.deg)
    assert "Rotated by" in wf_rot.history[-1]

    print("Validation successful!")

if __name__ == "__main__":
    test_poppy_features()
