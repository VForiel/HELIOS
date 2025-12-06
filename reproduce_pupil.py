
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import helios

def reproduce():
    print("--- Reproducing Pupil Issue ---")
    
    # User snippet
    wf = helios.Wavefront(wavelength=550*u.nm, size=100)
    wf.pixel_scale = 0.001 * u.m # 1 mm/pixel -> 10 cm total size
    print(f"Wavefront Pixel Scale: {wf.pixel_scale}")
    print(f"Wavefront Total Size: {wf.pixel_scale * wf.field.shape[-1]}")
    
    pup = helios.Pupil(diameter=0.05*u.m) # 5 cm diameter (half of grid)
    pup.add_disk(radius=0.025*u.m) # Half pupil
    
    print(f"Pupil Diameter: {pup.diameter} m")
    
    wf2 = pup.process(wf)
    
    intensity = np.abs(wf2.field[0])**2
    print(f"Intensity Min: {np.min(intensity)}")
    print(f"Intensity Max: {np.max(intensity)}")
    print(f"Intensity Mean: {np.mean(intensity)}")
    
    if np.allclose(intensity, 1.0):
        print("ISSUE REPRODUCED: Intensity is constant 1.0 everywhere.")
    else:
        print("Intensity is NOT constant.")

if __name__ == "__main__":
    reproduce()
