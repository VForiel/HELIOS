"""Diagnostic script to visualize input field distribution at z=0."""
import numpy as np
import matplotlib.pyplot as plt

# Parameters from warning
W = 10e-6  # 10 µm
N = 2
Sin = (W / N) / 4  # Default calculation

print(f"MMI width W = {W*1e6:.1f} µm")
print(f"Number of inputs N = {N}")
print(f"Input mode width Sin = {Sin*1e6:.3f} µm")
print(f"Gaussian sigma = {Sin/2*1e6:.3f} µm")
print(f"3*sigma extent = {3*Sin/2*1e6:.3f} µm")
print()

# Compute input positions (symmetric)
def _compute_symmetric_port_positions(N, W, D):
    """Simplified version of the function."""
    if D is None:
        D = W / (N + 1)
    positions = [W / (N + 1) * (i + 1) for i in range(N)]
    return np.array(positions)

input_positions = _compute_symmetric_port_positions(N, W, None)
print(f"Input positions: {input_positions*1e6} µm")
print()

# Create x grid (same as simulation)
x_grid = np.linspace(-W/2, 3*W/2, 500)
dx = x_grid[1] - x_grid[0]

# Construct input field (simplified)
def gaussian_mode(x, center, width):
    sigma = width / 2.0
    profile = np.exp(-((x - center)**2) / (sigma**2))
    norm_factor = np.sqrt(np.sum(np.abs(profile)**2) * dx)
    return profile / norm_factor

# Equal amplitude inputs
input_field = np.zeros_like(x_grid, dtype=complex)
for center in input_positions:
    input_field += gaussian_mode(x_grid, center, Sin)

# Compute intensities
intensity = np.abs(input_field)**2

# Define MMI core region
core_mask = (x_grid >= 0.5*W) & (x_grid <= 1.5*W)
core_min = 0.5 * W
core_max = 1.5 * W

# Integrate powers
power_core = np.sum(intensity[core_mask]) * dx
power_total = np.sum(intensity) * dx
fraction_outside = (power_total - power_core) / power_total

print(f"Power in core [5-15 µm]: {power_core:.3e}")
print(f"Power total: {power_total:.3e}")
print(f"Fraction outside core: {fraction_outside*100:.1f}%")
print()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel 1: Intensity profile
ax1 = axes[0]
ax1.plot(x_grid*1e6, intensity, 'b-', lw=2, label='Total intensity')
ax1.axvspan(core_min*1e6, core_max*1e6, alpha=0.2, color='green', label='MMI core [0.5W, 1.5W]')
ax1.axvline(0, color='red', linestyle='--', alpha=0.5, label='MMI boundaries [0, W]')
ax1.axvline(W*1e6, color='red', linestyle='--', alpha=0.5)

# Mark input positions
for center in input_positions:
    ax1.axvline(center*1e6, color='orange', linestyle=':', alpha=0.7)
    ax1.annotate(f'{center*1e6:.1f} µm', xy=(center*1e6, ax1.get_ylim()[1]*0.9), ha='center', fontsize=9)

ax1.set_xlabel('x [µm]', fontsize=12)
ax1.set_ylabel('Intensity [a.u.]', fontsize=12)
ax1.set_title(f'Input Field Intensity at z=0 (Sin={Sin*1e6:.3f} µm)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Cumulative power
cumulative_power = np.cumsum(intensity) * dx
ax2 = axes[1]
ax2.plot(x_grid*1e6, cumulative_power, 'g-', lw=2)
ax2.axhline(power_core, color='blue', linestyle='--', lw=2, label=f'Core power = {power_core:.2f}')
ax2.axhline(power_total, color='red', linestyle='--', lw=2, label=f'Total power = {power_total:.2f}')
ax2.axvspan(core_min*1e6, core_max*1e6, alpha=0.2, color='green')
ax2.axvline(core_min*1e6, color='blue', linestyle=':', alpha=0.5, label='Core boundaries')
ax2.axvline(core_max*1e6, color='blue', linestyle=':', alpha=0.5)

ax2.set_xlabel('x [µm]', fontsize=12)
ax2.set_ylabel('Cumulative Power', fontsize=12)
ax2.set_title('Cumulative Power Integration', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d:/HELIOS/tmp/input_field_diagnostic.png', dpi=150)
print(f"✓ Plot saved to d:/HELIOS/tmp/input_field_diagnostic.png")
plt.show()

print("\n" + "="*70)
print("DIAGNOSTIC CONCLUSION:")
print("="*70)
print(f"The default Sin = (W/N)/4 = {Sin*1e6:.3f} µm creates Gaussian modes")
print(f"with σ = {Sin/2*1e6:.3f} µm. These modes extend ±3σ ≈ ±{3*Sin/2*1e6:.1f} µm")
print(f"from their centers at {input_positions*1e6} µm.")
print()
print("With input positions near the MMI edges, the Gaussian tails extend")
print("significantly outside the core region [5, 15] µm, causing ~50% power loss.")
print()
print("SOLUTION:")
print("1. Reduce Sin (narrower Gaussian modes) → Less tail spreading")
print("2. Use MFD calculation from Marcuse formula (accounts for waveguide physics)")
print("3. Clip Gaussian tails at core boundaries (artificial but practical)")
print("="*70)
