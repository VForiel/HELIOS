import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import helios
from helios.core.simulation import Wavefront
from helios.core.context import Context
from helios.components.scene import Scene, Star, Planet
from helios.components.atmosphere import Atmosphere

def test_wavefront_structure():
    print("Testing Wavefront structure...")
    wf = Wavefront(wavelength=550*u.nm, size=128, samples=5)
    assert wf.field.shape == (5, 128, 128)
    assert wf.source_directions is None
    wf.source_directions = np.zeros((5, 2))
    assert wf.source_directions.shape == (5, 2)
    print("✓ Wavefront structure correct")

def test_context_coherent_sources():
    print("Testing Context coherent sources...")
    ctx = Context()
    scene = Scene()
    scene.add(Star(position=(0*u.arcsec, 0*u.arcsec)))
    scene.add(Planet(position=(1*u.arcsec, 0*u.arcsec)))
    ctx.add_layer(scene)
    
    wf = ctx.get_input_wavefront(wavelength=550*u.nm, size=128, coherent_sources=True)
    assert wf.field.shape[0] == 2
    print(f"✓ Created {wf.field.shape[0]} samples for 2 sources")
    
    # Check directions
    # Star at 0,0
    # Planet at 1,0 arcsec -> rad
    # 1 arcsec = 4.848e-6 rad
    print(f"Directions: {wf.source_directions}")

def test_context_grid_sampling():
    print("Testing Context grid sampling...")
    ctx = Context()
    scene = Scene()
    scene.add(Star())
    ctx.add_layer(scene)
    
    samples_1d = 3
    wf = ctx.get_input_wavefront(wavelength=550*u.nm, size=128, 
                                 angular_samples=samples_1d, coherent_sources=False)
    assert wf.field.shape[0] == samples_1d**2
    print(f"✓ Created {wf.field.shape[0]} samples for {samples_1d}x{samples_1d} grid")

def test_atmosphere_process():
    print("Testing Atmosphere process...")
    atm = Atmosphere(rms=100*u.nm)
    wf = Wavefront(wavelength=550*u.nm, size=128, samples=4)
    wf.field[:] = 1.0
    
    wf_out = atm.process(wf)
    assert wf_out.field.shape == (4, 128, 128)
    # Check if phase is applied
    phase = np.angle(wf_out.field)
    assert np.std(phase) > 0
    print("✓ Atmosphere applied phase to 3D wavefront")

def test_plotting():
    print("Testing Plotting...")
    wf = Wavefront(wavelength=550*u.nm, size=64, samples=4)
    # Add some phase
    wf.field = wf.field * np.exp(1j * np.random.rand(4, 64, 64))
    
    try:
        fig, ax = wf.plot(stack_method=np.mean)
        plt.close(fig)
        print("✓ Plot with stack_method=np.mean successful")
    except Exception as e:
        print(f"✗ Plot failed: {e}")
        raise e

if __name__ == "__main__":
    test_wavefront_structure()
    test_context_coherent_sources()
    test_context_grid_sampling()
    test_atmosphere_process()
    test_plotting()
