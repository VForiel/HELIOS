
import numpy as np
from helios.sim import mmi
import matplotlib.pyplot as plt

def sweep_length():
    W = 20.0e-6
    n_eff = 2.0458
    wavelength = 1.55e-6
    
    # L_pi
    L_pi = 4 * n_eff * W**2 / (3 * wavelength)
    print(f"L_pi = {L_pi*1e6:.2f} um")
    
    # Sweep around 3/4 L_pi and other fractions
    # 0 to L_pi
    lengths = np.linspace(10e-6, L_pi, 50)
    max_intensities = []
    
    input_vec = np.array([1, 1, 1, 1], dtype=complex)
    input_vec /= 2.0 # Normalize energy
    
    print("Sweeping Length for [1,1,1,1] focusing...")
    
    for L in lengths:
        # We need output vector, so output_file=None
        # But wait, we need to manually compute output vector if simulation doesn't return it? 
        # I updated mmi to return it!
        out = mmi(N=4, M=4, L=L, W=W, n_eff=n_eff, wavelength=wavelength, 
                 input_amplitudes=input_vec, output_file=None, num_modes=40)
        
        # Max intensity in any port
        intensities = np.abs(out)**2
        max_intensities.append(np.max(intensities))
        
    best_idx = np.argmax(max_intensities)
    best_L = lengths[best_idx]
    best_val = max_intensities[best_idx]
    
    print(f"\nBest Focusing for [1,1,1,1]:")
    print(f"L = {best_L*1e6:.2f} um")
    print(f"Max Port Intensity = {best_val:.3f}")
    
    # Check standard points
    points = {
        "1/4 L_pi": L_pi/4,
        "1/2 L_pi": L_pi/2,
        "3/4 L_pi": 3*L_pi/4,
        "1 L_pi": L_pi
    }
    
    print("\nCheck Standard Lengths:")
    for name, L_val in points.items():
        out = mmi(N=4, M=4, L=L_val, W=W, n_eff=n_eff, input_amplitudes=input_vec, output_file=None)
        val = np.max(np.abs(out)**2)
        print(f"{name} ({L_val*1e6:.1f}um) -> Max Int: {val:.3f}")

if __name__ == "__main__":
    sweep_length()
