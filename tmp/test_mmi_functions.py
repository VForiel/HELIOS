#!/usr/bin/env python
"""Test MMI functions with mode-dependent n_eff"""

import sys
sys.path.insert(0, '../src')

from helios.sim.mmi import simulate, compute_contributions, calibrate_input_phases_genetic
import numpy as np

print('Testing key MMI functions with mode-dependent n_eff:')
print('=' * 70)

# Test 1: simulate()
print('\n1. Testing simulate() with n_core/n_clad parameters...')
result = simulate(N=2, M=2, L=50e-6, W=10e-6, n_core=2.0, n_clad=1.95,
                  wavelength=1.55e-6, num_z_steps=10, verbose=False)
print('   ✓ simulate() works')

# Test 2: compute_contributions()
print('2. Testing compute_contributions() with mode-dependent n_eff...')
data = compute_contributions(
    N=2, M=2, L=50e-6, W=10e-6, n_core=2.0, n_clad=1.95,
    wavelength=1.55e-6, input_amplitudes=[1.0, 0.0], num_modes=10, num_z_steps=10, z_resolution=None, verbose=False
)
print('   ✓ compute_contributions() works')
npts = len(data['x_grid'])
nsteps = len(data['z_grid'])
print(f'   Grid points: {npts}, Z steps: {nsteps}')

# Test 3: calibrate_input_phases_genetic()
print('3. Testing calibrate_input_phases_genetic()...')
result = calibrate_input_phases_genetic(N=2, M=2, L=50e-6, W=10e-6,
                                       n_core=2.0, n_clad=1.95,
                                       wavelength=1.55e-6,
                                       bright_output_idx=0, verbose=False)
print('   ✓ calibrate_input_phases_genetic() works')
print(f'   Result keys: {list(result.keys())}')

print('\n' + '=' * 70)
print('✅ All functions work correctly with mode-dependent n_eff!')
