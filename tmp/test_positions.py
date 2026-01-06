#!/usr/bin/env python
"""Debug script to check input positions for N inputs."""

import numpy as np
import sys
sys.path.insert(0, r"D:\HELIOS\src")

from helios.sim.mmi import simulate

# Test with 4 inputs
print("\n" + "="*70)
print("TEST: 4 inputs, 2 outputs")
print("="*70)

result = simulate(
    N=4,
    M=2,
    L=100e-6,
    W=10e-6,
    n_core=2.0458,
    delta_n=0.0958,
    wavelength=1.55e-6,
    input_amplitudes=[1.0, 1.0, 1.0, 1.0],
    num_modes=30,
    num_z_steps=50,
    verbose=True
)

print(f"\nOutput amplitudes: {result}")
print(f"Output powers: {np.abs(result)**2}")
