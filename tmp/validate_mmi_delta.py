import numpy as np
from helios.sim.mmi import simulate, _compute_mmi_field, _solve_slab_modes_fd

wavelength = 1.55e-6
n_core = 2.0458
W = 10.0e-6
L = None
N = 2
M = 2
Sin = 2.5e-6
Sout = 2.5e-6
num_modes = 20
num_z_steps = 80
z_resolution = None
Din = 5.0e-6
Dout = 5.0e-6
input_amplitudes = np.sqrt(0.5) * np.array([1, 1j], dtype=complex)

print("Validation sweep on Δn")
print("Δn\tfirst n_eff (top 4)\toutputs |E|^2 (M=2)\ttotal power")

for delta_n in [0.0, 0.01, 0.0958]:
    # Auto L as in simulate
    n_clad = n_core - delta_n
    n_eff_guess = 0.7 * n_core + 0.3 * n_clad
    L_use = L
    if L_use is None:
        L_pi = 4 * n_eff_guess * W**2 / (3 * wavelength)
        L_use = L_pi / 2

    # Use the FD mode solver directly to inspect n_eff of guided modes
    x_grid = np.linspace(-W/2, 3*W/2, 500)
    n_profile = np.where((x_grid >= 0) & (x_grid <= W), n_core, n_clad)
    k0 = 2 * np.pi / wavelength
    betas, _ = _solve_slab_modes_fd(x_grid, n_profile, k0, num_modes)
    n_eff_est = float(betas[0] / k0) if len(betas) > 0 and k0 != 0 else np.nan

    # Run simulate to get outputs
    outputs = simulate(
        N=N, M=M, L=L_use, W=W, n_core=n_core, delta_n=delta_n, wavelength=wavelength,
        input_amplitudes=input_amplitudes, num_modes=num_modes, num_z_steps=num_z_steps,
        z_resolution=z_resolution, output_file=None, verbose=False, Din=Din, Dout=Dout,
        Sin=Sin, Sout=Sout,
    )
    intensities = np.abs(outputs) ** 2
    total_power = intensities.sum()
    print(f"{delta_n:.5f}\t{n_eff_est:.6f}\t[{intensities[0]:.4e}, {intensities[1]:.4e}]\t{total_power:.4e}")
