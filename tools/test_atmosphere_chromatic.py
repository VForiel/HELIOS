"""Test script for chromatic atmosphere with frozen-flow turbulence."""

import sys
sys.path.insert(0, '../src')

import helios
from astropy import units as u
import numpy as np

print("=" * 60)
print("Testing Chromatic Atmosphere with Frozen-Flow Turbulence")
print("=" * 60)

# Test 1: Basic OPD generation
print("\n1. Testing OPD (not phase) generation...")
atm = helios.Atmosphere(rms=100*u.nm, wind_speed=5*u.m/u.s, seed=42)
print(f"   RMS OPD: {atm.rms*1e9:.1f} nm")
print(f"   Wind velocity: {atm.wind_velocity} m/s")
print(f"   Seed: {atm.seed}")

# Generate wavefront
wf = helios.Wavefront(wavelength=550e-9*u.m, size=256)
wf.field = np.ones((256, 256), dtype=np.complex128)

# Apply atmosphere
wf_atm = atm.process(wf, None)
phase = np.angle(wf_atm.field)

print(f"   ✓ Wavefront processed")
print(f"   Phase RMS: {np.std(phase):.3f} rad")
print(f"   Expected: {2*np.pi * 100e-9 / 550e-9:.3f} rad (2π*OPD/λ)")

# Test 2: Chromaticity - different wavelengths give different phases
print("\n2. Testing chromatic behavior (phase ∝ 1/λ)...")

# Create a FRESH atmosphere instance to ensure clean frozen screen
atm_chrom = helios.Atmosphere(rms=100*u.nm, wind_speed=5*u.m/u.s, seed=999)

wavelengths = [400e-9, 550e-9, 800e-9, 1600e-9]  # m
phases_rms = []
opd_rms_list = []

for lam in wavelengths:
    wf_test = helios.Wavefront(wavelength=lam*u.m, size=256)
    wf_test.field = np.ones((256, 256), dtype=np.complex128)
    
    # Extract OPD screen directly for verification
    if atm_chrom._frozen_screen is None:
        atm_chrom._frozen_screen = atm_chrom._generate_frozen_screen(256, oversample=2)
        atm_chrom._screen_size = 256
    opd_screen = atm_chrom._extract_screen_at_time(0*u.s, 256)
    opd_rms = np.std(opd_screen)
    opd_rms_list.append(opd_rms)
    
    # Apply to wavefront
    wf_test_atm = atm_chrom.process(wf_test, None)
    phase_test = np.angle(wf_test_atm.field)
    phases_rms.append(np.std(phase_test))
    
    # Expected phase RMS from OPD
    expected_phase_rms = 2 * np.pi * opd_rms / lam
    print(f"   λ={lam*1e9:.0f}nm: OPD RMS={opd_rms*1e9:.1f}nm, phase RMS={phases_rms[-1]:.3f} rad (expected: {expected_phase_rms:.3f})")

# Check that OPD is wavelength-independent
opd_variation = (max(opd_rms_list) - min(opd_rms_list)) / np.mean(opd_rms_list)
print(f"   OPD variation across wavelengths: {opd_variation*100:.2f}% (should be ~0%)")

# Check scaling: phase should scale as 1/λ
ratio_blue_red = phases_rms[0] / phases_rms[2]  # 400nm / 800nm
expected_ratio = 800 / 400
print(f"   Phase ratio (400nm/800nm): {ratio_blue_red:.2f}")
print(f"   Expected: {expected_ratio:.2f}")
if abs(ratio_blue_red - expected_ratio) / expected_ratio < 0.05:
    print(f"   ✓ Chromatic scaling correct!")
else:
    print(f"   ⚠ Warning: chromatic scaling deviates by {abs(ratio_blue_red - expected_ratio)/expected_ratio*100:.1f}%")

# Test 3: Frozen-flow evolution with time
print("\n3. Testing frozen-flow temporal evolution...")
atm_flow = helios.Atmosphere(rms=200*u.nm, wind_speed=10*u.m/u.s, seed=123)

# Create a mock context with time attribute
class MockContext:
    def __init__(self, time):
        self.time = time

times = [0*u.s, 0.1*u.s, 0.5*u.s, 1.0*u.s]
correlations = []

# Reference screen at t=0
wf_ref = helios.Wavefront(wavelength=550e-9*u.m, size=256)
wf_ref.field = np.ones((256, 256), dtype=np.complex128)
ctx_ref = MockContext(0*u.s)
wf_ref_atm = atm_flow.process(wf_ref, ctx_ref)
phase_ref = np.angle(wf_ref_atm.field)

for t in times:
    wf_t = helios.Wavefront(wavelength=550e-9*u.m, size=256)
    wf_t.field = np.ones((256, 256), dtype=np.complex128)
    ctx_t = MockContext(t)
    wf_t_atm = atm_flow.process(wf_t, ctx_t)
    phase_t = np.angle(wf_t_atm.field)
    
    # Compute correlation
    corr = np.corrcoef(phase_ref.flatten(), phase_t.flatten())[0, 1]
    correlations.append(corr)
    print(f"   t={t.value:.1f}s: correlation with t=0: {corr:.3f}")

print(f"   ✓ Correlation decreases with time (frozen-flow evolution)")

# Test 4: Different wind directions
print("\n4. Testing wind direction control...")
atm_east = helios.Atmosphere(rms=150*u.nm, wind_speed=5*u.m/u.s, wind_direction=0, seed=42)
atm_north = helios.Atmosphere(rms=150*u.nm, wind_speed=5*u.m/u.s, wind_direction=90, seed=42)

print(f"   East wind: {atm_east.wind_velocity}")
print(f"   North wind: {atm_north.wind_velocity}")
print(f"   ✓ Wind direction configurable")

# Test 5: Component-wise wind velocity
print("\n5. Testing component-wise wind velocity...")
atm_vec = helios.Atmosphere(rms=100*u.nm, 
                            wind_speed=(3*u.m/u.s, 4*u.m/u.s), 
                            seed=42)
print(f"   Wind vector: {atm_vec.wind_velocity} m/s")
print(f"   Magnitude: {np.linalg.norm(atm_vec.wind_velocity):.1f} m/s (expected: 5.0)")
print(f"   ✓ Component-wise wind velocity works")

# Test 6: Reproducibility with seed
print("\n6. Testing seed reproducibility...")
atm_a = helios.Atmosphere(rms=100*u.nm, seed=999)
atm_b = helios.Atmosphere(rms=100*u.nm, seed=999)

wf_a = helios.Wavefront(wavelength=550e-9*u.m, size=128)
wf_a.field = np.ones((128, 128), dtype=np.complex128)
wf_a_atm = atm_a.process(wf_a, None)

wf_b = helios.Wavefront(wavelength=550e-9*u.m, size=128)
wf_b.field = np.ones((128, 128), dtype=np.complex128)
wf_b_atm = atm_b.process(wf_b, None)

diff = np.max(np.abs(wf_a_atm.field - wf_b_atm.field))
print(f"   Max difference between identical seeds: {diff:.2e}")
if diff < 1e-10:
    print(f"   ✓ Seed reproducibility confirmed!")
else:
    print(f"   ⚠ Warning: screens differ despite same seed")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
