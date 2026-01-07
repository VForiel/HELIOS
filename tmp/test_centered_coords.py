"""Diagnostic script with CORRECTED centered coordinate system."""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
W = 10e-6  # 10 µm
N = 2
Sin = (W / N) / 4  # Default calculation

print("="*70)
print("CENTERED COORDINATE SYSTEM VALIDATION")
print("="*70)
print(f"MMI width W = {W*1e6:.1f} µm")
print(f"MMI core region: [-W/2, W/2] = [{-W/2*1e6:.1f}, {W/2*1e6:.1f}] µm")
print(f"Simulation window: [-W, W] = [{-W*1e6:.1f}, {W*1e6:.1f}] µm")
print(f"Number of inputs N = {N}")
print(f"Input mode width Sin = {Sin*1e6:.3f} µm")
print(f"Gaussian sigma = {Sin/2*1e6:.3f} µm")
print()

# Compute input positions (centered at x=0)
def compute_centered_positions(num_ports, W, spacing=None):
    """Centered coordinate system."""
    if spacing is None:
        spacing = W / num_ports
    center = 0.0  # Centered at x=0
    offsets = (np.arange(num_ports, dtype=float) - 0.5 * (num_ports - 1)) * spacing
    positions = center + offsets
    return positions

input_positions = compute_centered_positions(N, W, spacing=None)
print(f"Input positions (centered): {input_positions*1e6} µm")
print()

# Create x grid (centered)
x_grid = np.linspace(-W, W, 500)
dx = x_grid[1] - x_grid[0]

# Construct input field
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

# Define MMI core region (CENTERED)
core_mask = (x_grid >= -W/2) & (x_grid <= W/2)
core_min = -W/2
core_max = W/2

# Integrate powers
power_core = np.sum(intensity[core_mask]) * dx
power_total = np.sum(intensity) * dx
fraction_outside = (power_total - power_core) / power_total

print("POWER DISTRIBUTION:")
print("-" * 70)
print(f"Power in core [-5, +5] µm: {power_core:.3e}")
print(f"Power total: {power_total:.3e}")
print(f"Fraction outside core: {fraction_outside*100:.1f}%")
print()

# Determine if acceptable
if fraction_outside < 0.10:
    status = "✅ PASS"
else:
    status = "❌ FAIL"
print(f"{status}: Power outside core = {fraction_outside*100:.1f}% (target: <10%)")
print()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel 1: Intensity profile
ax1 = axes[0]
ax1.plot(x_grid*1e6, intensity, 'b-', lw=2, label='Total intensity')
ax1.axvspan(core_min*1e6, core_max*1e6, alpha=0.2, color='green', label='MMI core [-W/2, W/2]')
ax1.axvline(0, color='black', linestyle='-', lw=2, alpha=0.7, label='MMI center (x=0)')

# Mark input positions
for center in input_positions:
    ax1.axvline(center*1e6, color='orange', linestyle=':', alpha=0.7, lw=1.5)
    ax1.annotate(f'{center*1e6:.1f} µm', xy=(center*1e6, ax1.get_ylim()[1]*0.9), ha='center', fontsize=9)

ax1.set_xlabel('x [µm]', fontsize=12)
ax1.set_ylabel('Intensity [a.u.]', fontsize=12)
ax1.set_title(f'Input Field Intensity at z=0 (CENTERED COORDINATES)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-W*1e6, W*1e6)

# Panel 2: Cumulative power
cumulative_power = np.cumsum(intensity) * dx
ax2 = axes[1]
ax2.plot(x_grid*1e6, cumulative_power, 'g-', lw=2)
ax2.axhline(power_core, color='blue', linestyle='--', lw=2, label=f'Core power = {power_core:.2f}')
ax2.axhline(power_total, color='red', linestyle='--', lw=2, label=f'Total power = {power_total:.2f}')
ax2.axvspan(core_min*1e6, core_max*1e6, alpha=0.2, color='green')
ax2.axvline(core_min*1e6, color='blue', linestyle=':', alpha=0.5, label='Core boundaries')
ax2.axvline(core_max*1e6, color='blue', linestyle=':', alpha=0.5)
ax2.axvline(0, color='black', linestyle='-', lw=2, alpha=0.7)

ax2.set_xlabel('x [µm]', fontsize=12)
ax2.set_ylabel('Cumulative Power', fontsize=12)
ax2.set_title('Cumulative Power Integration (CENTERED)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-W*1e6, W*1e6)

plt.tight_layout()
plt.savefig('d:/HELIOS/tmp/centered_coords_validation.png', dpi=150)
print(f"✓ Plot saved to d:/HELIOS/tmp/centered_coords_validation.png")
plt.show()

print("\n" + "="*70)
print("VALIDATION CONCLUSION:")
print("="*70)
print("With CENTERED coordinate system:")
print(f"- MMI core: [-W/2, W/2] = [-5, +5] µm")
print(f"- Input positions: {input_positions*1e6} µm")
print(f"- Both inputs are NOW WITHIN the core region!")
print()
print(f"Power outside core: {fraction_outside*100:.1f}%")
if fraction_outside < 0.10:
    print("✅ This is acceptable (<10% threshold)")
    print("The coordinate system is now CONSISTENT.")
else:
    print(f"⚠️  Still {fraction_outside*100:.1f}% outside - Sin may need to be reduced further")
print("="*70)
