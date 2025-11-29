import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios
from astropy import units as u

def test_full_simulation_flow():
    # Scene
    scene = helios.components.Scene(distance=10*u.pc)
    scene.add(helios.components.Star(temperature=5000*u.K, magnitude=5, position=(0*u.AU, 0*u.AU)))
    
    # Optics
    telescope = helios.components.TelescopeArray(latitude=0*u.deg, longitude=0*u.deg, altitude=2000*u.m)
    telescope.add_collector(pupil=helios.components.Pupil(), position=(0,0), size=8*u.m)
    
    # Detector
    camera = helios.components.Camera(pixels=(10, 10))
    
    # Context
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)
    
    # Run
    result = ctx.observe()
    assert result.shape == (10, 10)

if __name__ == "__main__":
    test_full_simulation_flow()
    print("Integration test passed.")
