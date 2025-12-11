import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from helios.core.pipeline import Pipeline, Element
from helios.components.collector import TelescopeArray, Collector
from helios.components.scene import Scene, Star

from helios.components.pupil import Pupil

def test_interferometer_phase_generation():
    print("Testing Interferometer Phase Generation (Piston + Tilt)...")
    
    # 1. Setup Pipeline with an off-axis source
    # Source at theta_x = 1 arcsec, theta_y = 0
    theta_x = 1.0 * u.arcsec
    theta_y = 0.0 * u.arcsec
    wavelength = 1.0 * u.um
    
    star = Star(position=(theta_x, theta_y), magnitude=0)
    scene = Scene()
    scene.add(star)
    
    # Pipeline parameters
    npix = 128
    diameter = 1.0 * u.m # Collector diameter
    
    pipe = Pipeline(layers=[scene], wavelength=wavelength, npix=npix, diameter=diameter)
    
    # 2. Setup TelescopeArray with 2 collectors
    # Baseline B = 100m along x-axis
    # C1 at (-50, 0), C2 at (+50, 0)
    pupil = Pupil(diameter=diameter)
    pupil.add_disk(radius=diameter/2, center=(0.0*u.m, 0.0*u.m))
    
    c1 = Collector(pupil=pupil, position=(-50.0, 0.0), size=diameter)
    c2 = Collector(pupil=pupil, position=(50.0, 0.0), size=diameter)
    
    ta = TelescopeArray()
    ta.add_element(c1)
    ta.add_element(c2)
    
    # 3. Process (should trigger get_input_wavefront with collectors)
    # Passing None as wavefront to trigger generation
    wf_array = ta.process(None, pipe)
    
    # 4. Verifications
    
    # A. Check type and length
    if type(wf_array).__name__ != 'WavefrontArray':
        print(f"FAILURE: Expected WavefrontArray, got {type(wf_array)}")
        return
    
    if len(wf_array) != 2:
        print(f"FAILURE: Expected 2 wavefronts, got {len(wf_array)}")
        return
        
    print("SUCCESS: Returned WavefrontArray with 2 wavefronts.")
    
    # Constants
    k = 2 * np.pi / wavelength.to(u.m).value
    tx_rad = theta_x.to(u.rad).value
    
    # B. Verify Phase for Collector 1 (Left, x = -50)
    wf1 = wf_array[0]
    field1 = wf1[0] # Source 0
    phase1 = np.angle(field1)
    
    # Expected Piston 1: k * (cx * tx)
    # cx = -50
    expected_piston1 = k * (-50.0 * tx_rad)
    # Wrap to [-pi, pi]
    expected_piston1 = (expected_piston1 + np.pi) % (2 * np.pi) - np.pi
    
    # Measure phase at center (approx)
    center_idx = npix // 2
    measured_phase1 = phase1[center_idx, center_idx]
    
    print(f"\nCollector 1 (-50m):")
    print(f"  Expected Piston Phase: {expected_piston1:.4f} rad")
    print(f"  Measured Center Phase: {measured_phase1:.4f} rad")
    
    if not np.isclose(measured_phase1.to(u.rad).value, expected_piston1, atol=1e-3):
        print("  FAILURE: Piston mismatch for C1")
    else:
        print("  SUCCESS: Piston matches for C1")

    # C. Verify Phase for Collector 2 (Right, x = +50)
    wf2 = wf_array[1]
    field2 = wf2[0]
    phase2 = np.angle(field2)
    
    # Expected Piston 2: k * (cx * tx)
    # cx = +50
    expected_piston2 = k * (50.0 * tx_rad)
    # Wrap to [-pi, pi]
    expected_piston2 = (expected_piston2 + np.pi) % (2 * np.pi) - np.pi
    
    measured_phase2 = phase2[center_idx, center_idx]
    
    print(f"\nCollector 2 (+50m):")
    print(f"  Expected Piston Phase: {expected_piston2:.4f} rad")
    print(f"  Measured Center Phase: {measured_phase2:.4f} rad")
    
    if not np.isclose(measured_phase2.to(u.rad).value, expected_piston2, atol=1e-3):
        print("  FAILURE: Piston mismatch for C2")
    else:
        print("  SUCCESS: Piston matches for C2")
        
    # D. Verify Differential Piston (OPD)
    # Delta Phi = k * B * theta
    # B = 100
    expected_dphi = k * (100.0 * tx_rad)
    expected_dphi_wrapped = (expected_dphi + np.pi) % (2 * np.pi) - np.pi
    
    measured_dphi = measured_phase2 - measured_phase1
    measured_dphi_wrapped = (measured_dphi + np.pi*u.rad) % (2 * np.pi*u.rad) - np.pi*u.rad
    
    print(f"\nDifferential Phase (C2 - C1):")
    print(f"  Expected Delta Phi: {expected_dphi_wrapped:.4f} rad")
    print(f"  Measured Delta Phi: {measured_dphi_wrapped:.4f}")
    
    if not np.isclose(measured_dphi_wrapped.to(u.rad).value, expected_dphi_wrapped, atol=1e-3):
        print("  FAILURE: Differential phase mismatch")
    else:
        print("  SUCCESS: Differential phase matches")

    # E. Verify Tilt (Gradient across aperture)
    # For both collectors, the tilt should be the same (same source angle)
    # Tilt slope = k * theta_x
    # Phase difference across diameter D = k * D * theta_x
    
    # Get phase at edges (u = -D/2 and u = +D/2)
    # In pixels: 0 and npix-1
    # Note: Pixel 0 is at -D/2, Pixel npix-1 is at +D/2 - dx
    # Let's use the coordinate grid to be precise
    
    # Reconstruct grid
    u_vec = np.linspace(-diameter.value/2, diameter.value/2, npix)
    
    # Pick two points inside the aperture to avoid edge effects (mask=0)
    x1_idx = npix // 4
    x2_idx = 3 * npix // 4
    u1 = u_vec[x1_idx]
    u2 = u_vec[x2_idx]
    du = u2 - u1
    
    expected_tilt_diff = k * du * tx_rad
    expected_tilt_diff = (expected_tilt_diff + np.pi) % (2 * np.pi) - np.pi
    
    # Measure on C1
    p1 = phase1[center_idx, x1_idx]
    p2 = phase1[center_idx, x2_idx]
    measured_tilt_diff = p2 - p1
    measured_tilt_diff = (measured_tilt_diff + np.pi*u.rad) % (2 * np.pi*u.rad) - np.pi*u.rad
    
    print(f"\nTilt Verification (across {du:.2f}m aperture):")
    print(f"  Expected Phase Diff: {expected_tilt_diff:.4f} rad")
    print(f"  Measured Phase Diff: {measured_tilt_diff:.4f}")
    
    if not np.isclose(measured_tilt_diff.to(u.rad).value, expected_tilt_diff, atol=1e-2):
        print("  FAILURE: Tilt mismatch")
    else:
        print("  SUCCESS: Tilt matches")

if __name__ == "__main__":
    test_interferometer_phase_generation()
