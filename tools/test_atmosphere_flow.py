import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios
from astropy import units as u
import numpy as np

def test_atmosphere_flow():
    print("Testing Atmosphere flow...")
    
    # Scene
    scene = helios.components.Scene(distance=10*u.pc)
    scene.add(helios.components.Star(temperature=5000*u.K, magnitude=5))
    
    # Atmosphere
    # Use large RMS to ensure phase is not zero
    atm = helios.components.Atmosphere(rms=500*u.nm, wind_speed=10*u.m/u.s, seed=42)
    
    # Optics
    telescope = helios.components.TelescopeArray()
    telescope.add_collector(pupil=helios.components.Pupil(), position=(0,0), size=8*u.m)
    
    # Context
    ctx = helios.Context(wavelength=550*u.nm, npix=128)
    ctx.add_layer(scene)
    ctx.add_layer(atm)
    ctx.add_layer(telescope)
    
    # Run
    # TelescopeArray returns WavefrontArray
    # We can inspect the result of observe() if we don't add a camera
    # But observe() returns the output of the last layer.
    
    result = ctx.observe()
    
    assert isinstance(result, helios.core.simulation.WavefrontArray)
    assert len(result) == 1
    wf = result[0]
    
    # Check if phase is non-zero (Atmosphere applied)
    phase = np.angle(wf.field)
    rms_phase = np.std(phase)
    print(f"Phase RMS: {rms_phase:.4f} rad")
    
    assert rms_phase > 0.1, "Atmosphere should introduce significant phase error"
    
    print("Atmosphere flow test passed.")

if __name__ == "__main__":
    test_atmosphere_flow()
