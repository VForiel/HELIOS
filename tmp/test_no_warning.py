"""Test final pour vérifier le warning de normalisation."""
from helios.sim.mmi import simulate
import numpy as np
import warnings

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # Run simulation with default parameters
    result = simulate(
        N=2, M=2,
        L=None,  # Auto
        W=10e-6,
        wavelength=1.55e-6,
        input_amplitudes=np.array([1.0, 1.0]) / np.sqrt(2),
        num_modes=50,
        verbose=True
    )
    
    # Check for warnings
    print("\n" + "="*70)
    print("WARNINGS CHECK:")
    print("="*70)
    if len(w) > 0:
        print(f"❌ {len(w)} warning(s) raised:")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")
    else:
        print("✅ No warnings raised!")
        print("   Power distribution at z=0 is physically correct.")
    print("="*70)
