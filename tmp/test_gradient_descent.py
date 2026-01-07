"""Test script for the new gradient descent n_core optimization"""
import sys
sys.path.insert(0, 'D:/HELIOS/src')

import numpy as np
from helios.sim.mmi import calibrate_n_core_and_phases

# Test parameters
N = 2
M = 2
n_core_initial = 2.0458
wavelength = 1.55e-6
input_amplitudes = np.array([1.0, 1.0], dtype=float)

print("="*70)
print("TESTING GRADIENT DESCENT N_CORE OPTIMIZATION")
print("="*70)

# Progress callbacks for testing
def callback_coarse(current, total):
    print(f"  Coarse: {current}/{total}")

def callback_gradient(iteration, delta_n):
    print(f"  Gradient iter {iteration}: Δn_core = {delta_n:.4f}")

print("\nRunning optimization...")
result = calibrate_n_core_and_phases(
    N=N,
    M=M,
    L=None,
    W=10.0e-6,
    n_core_initial=n_core_initial,
    n_core_min=1.0,
    n_core_max=2.0 * n_core_initial,
    delta_n=0.0958,
    wavelength=wavelength,
    input_amplitudes=input_amplitudes,
    bright_output_idx=0,
    num_modes=50,
    num_z_steps=30,
    n_core_steps_coarse=10,
    gradient_convergence_threshold=1e-3,
    gradient_initial_step=0.01,
    verbose=True,
    progress_callback_coarse=callback_coarse,
    progress_callback_gradient=callback_gradient
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Coarse scan: {len(result['n_core_values_coarse'])} points")
print(f"Gradient descent: {len(result['n_core_values_gradient'])} points ({len(result['n_core_values_gradient'])-1} iterations)")
print(f"Optimal n_core: {result['best_n_core']:.4f}")
print(f"Best metric: {result['best_metric']:.3e}")
print(f"Best phases: {result['best_phases']}")
print("="*70)

# Verify structure
assert 'n_core_values_coarse' in result
assert 'metrics_coarse' in result
assert 'n_core_values_gradient' in result
assert 'metrics_gradient' in result
assert 'best_n_core' in result
assert 'best_metric' in result
assert 'best_phases' in result

print("\n✓ All tests passed!")
