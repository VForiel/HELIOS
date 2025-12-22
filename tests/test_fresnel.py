
import numpy as np
from astropy import units as u
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

import helios
from helios.core.wavefront import PlaneType

def test_fresnel_propagation():
    print("Testing Fresnel Propagation (ASM)...")
    
    # 1. Create a simple wavefront (e.g. a square aperture)
    size = 10 * u.mm
    npix = 256
    wavelength = 633 * u.nm # HeNe laser
    
    wf = helios.Wavefront(wavelength=wavelength, size=size, npix=npix)
    
    # Create a square aperture
    pupil = helios.Pupil(diameter=size)
    # 2mm square hole
    pupil.add_disk(radius=1*u.mm) 
    wf[:] = pupil.get_array(npix)
    
    print(f"Initial Plane: {wf.planetype}")
    
    # 2. Propagate by a small distance (Talbot distance or just near field)
    # Talbot length Zt = 2 * a^2 / lambda for a grating, but here we just want to see diffraction rings
    z = 10 * u.cm
    
    wf_prop = wf.propagate_fresnel(distance=z)
    
    print(f"Propagated Plane: {wf_prop.planetype}")
    assert wf_prop.planetype == wf.planetype # Should remain PUPIL/INTERMEDIATE
    
    print(f"History: {wf_prop.history[-1]}")
    assert "Propagated Fresnel" in wf_prop.history[-1]
    
    # Check energy conservation (Parseval's theorem implies unitary transform for ASM)
    # Total intensity should be roughly conserved (minus evanescent waves)
    print(f"Initial Intensity: {wf.total_intensity}")
    print(f"Final Intensity: {wf_prop.total_intensity}")
    
    ratio = wf_prop.total_intensity / wf.total_intensity
    print(f"Energy Ratio: {ratio}")
    assert 0.9 < ratio < 1.1
    
    print("Validation successful!")

if __name__ == "__main__":
    test_fresnel_propagation()
