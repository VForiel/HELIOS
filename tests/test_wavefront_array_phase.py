
import sys
import os
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from helios.core.simulation import Wavefront, WavefrontArray
from helios.components.collector import TelescopeArray
from helios.components.pupil import Pupil
from helios.core.context import Context

def test_wavefront_array_plot_layout():
    print("Testing WavefrontArray.plot layout...")
    
    # Create WavefrontArray with 2 channels, each with 2 samples
    wf1 = Wavefront(wavelength=550*u.nm, size=32, samples=2)
    wf1.sources = ["Src 1", "Src 2"]
    wf2 = Wavefront(wavelength=550*u.nm, size=32, samples=2)
    wf2.sources = ["Src 1", "Src 2"]
    
    wa = WavefrontArray([wf1, wf2])
    
    # Mock plt.subplots
    with patch('matplotlib.pyplot.show') as mock_show, \
         patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.colorbar') as mock_colorbar:
         
        mock_fig = MagicMock()
        # Expect 2 samples * 2 channels = 4 rows
        # 3 columns (Amp, LogAmp, Phase)
        mock_axes = np.empty((4, 3), dtype=object)
        for i in range(4):
            for j in range(3):
                mock_axes[i, j] = MagicMock()
        
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        wa.plot(show=False, log_scale=True)
        
        # Verify subplots called with correct rows/cols
        args, kwargs = mock_subplots.call_args
        nrows, ncols = args
        print(f"  subplots called with nrows={nrows}, ncols={ncols}")
        assert nrows == 4
        assert ncols == 3
        print("  Layout correct.")

def test_telescope_array_phase_shift():
    print("Testing TelescopeArray phase shift...")
    
    # Create TelescopeArray with 2 collectors
    ta = TelescopeArray()
    pupil = Pupil(diameter=1*u.m)
    pupil.add_disk(radius=0.5*u.m)
    # Collector 1 at (0,0)
    ta.add_collector(pupil, position=(0,0), size=1*u.m)
    # Collector 2 at (100,0) meters
    ta.add_collector(pupil, position=(100,0), size=1*u.m)
    
    # Create input wavefront with off-axis source
    # Source at theta_x = 1e-6 rad (approx 0.2 arcsec)
    theta_x = 1e-6
    wf = Wavefront(wavelength=1*u.um, size=32, samples=1)
    wf.source_directions = np.array([[theta_x, 0.0]]) * u.rad
    
    # Process
    ctx = Context()
    wa = ta.process(wf, ctx)
    
    # Check phase of output wavefronts
    # Collector 1: (0,0) -> phase shift 0
    # Collector 2: (100,0) -> phase shift k * x * theta_x
    # k = 2pi / 1e-6 = 2pi * 1e6
    # shift = 2pi * 1e6 * 100 * 1e-6 = 200 pi
    # Wait, 200 pi is a multiple of 2pi, so phase is 0 (modulo 2pi).
    # Let's pick theta_x so shift is pi/2.
    # k * 100 * theta_x = pi/2
    # 2pi/lambda * 100 * theta_x = pi/2
    # 4 * 100 * theta_x / lambda = 1
    # theta_x = lambda / 400 = 1e-6 / 400 = 2.5e-9 rad
    
    theta_x_test = 2.5e-9
    wf_test = Wavefront(wavelength=1*u.um, size=32, samples=1)
    wf_test.source_directions = np.array([[theta_x_test, 0.0]]) * u.rad
    
    wa_test = ta.process(wf_test, ctx)
    
    # Get central phase (field is ones initially)
    # Collector 1
    field1 = wa_test.wavefronts[0].field[0, 16, 16]
    phase1 = np.angle(field1)
    print(f"  Collector 1 phase: {phase1:.4f} rad")
    
    # Collector 2
    field2 = wa_test.wavefronts[1].field[0, 16, 16]
    phase2 = np.angle(field2)
    print(f"  Collector 2 phase: {phase2:.4f} rad")
    
    # Expected phase difference
    # k * 100 * theta_x_test = 2pi/1e-6 * 100 * 2.5e-9 = 2pi * 100 * 0.0025 = 2pi * 0.25 = pi/2
    expected_diff = np.pi/2
    diff = (phase2 - phase1) % (2*np.pi)
    if diff > np.pi: diff -= 2*np.pi
    
    print(f"  Phase diff: {diff:.4f} rad, Expected: {expected_diff:.4f} rad")
    
    assert np.isclose(diff, expected_diff, atol=1e-3)
    print("  Phase shift correct.")

if __name__ == "__main__":
    test_wavefront_array_plot_layout()
    test_telescope_array_phase_shift()
    print("All tests passed!")
