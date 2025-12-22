
import numpy as np
from helios.sim import mmi

def get_mmi_matrix(N=4, M=4, L=None, W=10e-6, n_eff=2.0458, wavelength=1.55e-6):
    # Calculate L_pi
    L_pi = 4 * n_eff * W**2 / (3 * wavelength)
    
    if L is None:
        # Standard General Interference for NxN is 3 * L_pi / N
        L = 3 * L_pi / N
    
    print(f" Simulating 4x4 MMI. W={W*1e6:.2f}um, L={L*1e6:.2f}um")
    
    matrix = np.zeros((M, N), dtype=complex)
    
    # Extract columns by injecting unit amplitude in each input
    for i in range(N):
        inputs = np.zeros(N, dtype=complex)
        inputs[i] = 1.0
        outputs = mmi(N=N, M=M, L=L, W=W, n_eff=n_eff, wavelength=wavelength, 
                      input_amplitudes=inputs, output_file=None, num_modes=80)
        matrix[:, i] = outputs
        
    return matrix

# Target Matrix (Butler / DFT like) specified by user
# 0.5 * [[1, 1, 1, 1], [1, -1, 1, -1], [1, i, -1, -i], [1, -i, -1, i]]
# Note: User's matrix rows are:
# 0: all 1
# 1: 1, -1, 1, -1 (period 2)
# 2: 1, i, -1, -i (period 4)
# 3: 1, -i, -1, i (reverse period 4)

target = 0.5 * np.array([
    [1, 1, 1, 1],
    [1, -1, 1, -1],
    [1, 1j, -1, -1j],
    [1, -1j, -1, 1j]
])



# Simulate standard 4x4
test_W = 20e-6
# Try Paired Interference (L_pi / 2)
# L_pi = 4 * n * W^2 / (3 * lambda)
test_L = (4 * 2.0458 * test_W**2 / (3 * 1.55e-6)) / 2
print(f"Testing Paired Interference Length: {test_L*1e6:.2f} um")
H_params = {'L': test_L, 'W': test_W}
H = get_mmi_matrix(N=4, M=4, L=test_L, W=test_W)



print("\n--- Extracted MMI Matrix H (absolute values) ---")
print(np.abs(H))

# Normalize phases to first row (remove output phase offsets? No, simple relative phase)
# Let's normalize each column by its first element phase
phase_ref = H / np.exp(1j * np.angle(H[0, :])) # Make first row real positive (0 phase)
normalized_phases = np.angle(phase_ref) / np.pi

print("\n--- Column Phases (relative to row 0) / pi ---")
print(normalized_phases)

# Target phases signature:
# Col 1: [0, 0, 0, 0]
# Col 2: [0, 1, 0, 1] (modulo 2)
# Col 3: [0, 0.5, 1, 1.5]
# Col 4: [0, -0.5, 1, -1.5] -> [0, 1.5, 1, 0.5]

print("Target Columns (unordered):")
print("A: [0, 0, 0, 0] (Uniform)")
print("B: [0, 1, 0, 1] (Alternating)")
print("C: [0, 0.5, 1, 1.5] (Step pi/2)")
print("D: [0, -0.5, -1, -1.5] (Step -pi/2)")



print("\n--- Testing Butler Inputs Focusing ---")
vectors = {
    "Sum [1,1,1,1]": np.array([1, 1, 1, 1]),
    "Alt [1,-1,1,-1]": np.array([1, -1, 1, -1]),
    "Quad+ [1, 1j, -1, -1j]": np.array([1, 1j, -1, -1j]),
    "Quad- [1, -1j, -1, 1j]": np.array([1, -1j, -1, 1j])
}

for name, vec in vectors.items():
    # Normalize vector
    vec = vec / np.linalg.norm(vec)
    out = mmi(N=4, M=4, L=H_params['L'], W=H_params['W'], n_eff=2.0458, input_amplitudes=vec, output_file=None)
    intensities = np.abs(out)**2
    print(f"\n{name} -> Intensities: {np.round(intensities, 3)}")
    peak_idx = np.argmax(intensities)
    print(f"Peak at Output {peak_idx+1} with {intensities[peak_idx]:.3f} power")

