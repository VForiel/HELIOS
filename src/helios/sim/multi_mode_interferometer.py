
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

def simulate_mmi(N=2, M=2, L=None, W=None, n_eff=2.0458, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=200, output_file=None, verbose=False):
    """
    Simulates light propagation in an NxM MMI (Multi-Mode Interferometer) using Eigenmode Expansion (Hard Wall).
    
    Args:
        N (int): Number of input ports.
        M (int): Number of output ports.
        L (float, optional): Length of the MMI region [m]. If None, calculated for 2x2 Paired Interference.
        W (float, optional): Width of the MMI region [m]. If None, defaults to 10um.
        n_eff (float): Effective refractive index of the MMI slab.
        wavelength (float): Operating wavelength [m].
        input_amplitudes (list/array, optional): Complex amplitudes for the N inputs. 
                                                 Defaults to uniform [1/sqrt(N), ...].
        num_modes (int): Number of modes to use for the decomposition.
        num_z_steps (int): Number of steps for z-propagation in the animation/plot.
        output_file (str, optional): Filename for the output animation. If None, no animation is generated.
        verbose (bool): If True, prints detailed simulation steps. Defaults to False.

    Returns:
        np.array: Complex amplitudes at the M outputs.
    """
    
    # Defaults logic
    if W is None:
        W = 10.0e-6 # 10 um default width
        
    if L is None:
        # Calculate length for 2x2 Paired Interference Coupler (3dB)
        # L_pi = 4 * n * W^2 / (3 * lambda)
        # L_couple = L_pi / 2 for N=2 paired
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2
        if verbose:
            print(f"Auto-calculated L = {L*1e6:.2f} um for W = {W*1e6:.2f} um (Paired Interference)")

    if input_amplitudes is None:
        # Uniform, phase 0
        val = 1.0 / np.sqrt(N)
        input_amplitudes = [val] * N
    
    if len(input_amplitudes) != N:
        raise ValueError(f"Length of input_amplitudes ({len(input_amplitudes)}) must match N ({N})")

    k0 = 2 * np.pi / wavelength
    
    # ... (rest of the simulation setup is unchanged)
    
        # 1. Define Geometry and Input Positions (Self-Imaging)
    # Standard input positions for N inputs: W/N * (i - 1/2)
    # i range 1..N -> index 0..N-1: i+1 - 0.5 = i + 0.5
    input_positions = [W/N * (i + 0.5) for i in range(N)]
    
    # 2. Define Waveguide Modes (Hard Wall Approximation)
    # Mode profiles: phi_m(x) = sqrt(2/W) * sin((m+1)*pi*x/W)
    # Propagation constants: beta_m = sqrt((k0*n)^2 - ((m+1)*pi/W)^2)
    
    x_grid = np.linspace(0, W, 500)
    dx = x_grid[1] - x_grid[0]
    
    modes = []
    betas = []
    
    for m in range(num_modes):
        # Transverse wave vector
        kx_m = (m + 1) * np.pi / W
        
        # Propagation constant
        # Check for cutoff
        sq_term = (k0 * n_eff)**2 - kx_m**2
        if sq_term < 0:
            beta_m = 0 # Evanescent, ignore or set to 0 imaginary part for simplified propagation
        else:
            beta_m = np.sqrt(sq_term)
            
        betas.append(beta_m)
        
        # Mode profile
        phi_m = np.sqrt(2/W) * np.sin(kx_m * x_grid)
        modes.append(phi_m)
        
    betas = np.array(betas)
    modes = np.array(modes) # Shape: (num_modes, num_x_points)
    
    # 3. Construct Input Field (Gaussian beams)
    input_field = np.zeros_like(x_grid, dtype=complex)
    beam_waist = (W / N) / 4 # Heuristic
    
    if verbose:
        print(f"Injecting input vector: {input_amplitudes}")
    
    for idx, amp in enumerate(input_amplitudes):
        if amp == 0: continue
        center_x = input_positions[idx]
        # Gaussian profile for this port
        gauss = np.exp(-((x_grid - center_x)**2) / (beam_waist**2))
        
        # Normalize to unit energy
        norm_factor = np.sqrt(np.sum(np.abs(gauss)**2) * dx)
        gauss = gauss / norm_factor
        
        input_field += amp * gauss

    # 4. Mode Decomposition
    # c_m = integral(E_in * phi_m) dx
    coeffs = []
    for m in range(num_modes):
        c_m = np.sum(input_field * modes[m]) * dx
        coeffs.append(c_m)
    coeffs = np.array(coeffs)
    
    # 5. Propagation
    z_grid = np.linspace(0, L, num_z_steps)
    
    # Field E(x, z)
    # shape: (num_z, num_x)
    field_evolution = np.zeros((num_z_steps, len(x_grid)), dtype=complex)
    
    for iz, z in enumerate(z_grid):
        # E(x,z) = sum(c_m * phi_m(x) * exp(-j * beta_m * z))
        # Vectorized sum over modes
        phase_term = np.exp(-1j * betas * z) # shape (num_modes,)
        
        # modes has shape (num_modes, num_x)
        # We want sum_m (c_m * phase_m * mode_m(x))
        # weight_m = c_m * phase_term
        weights = coeffs * phase_term # shape (num_modes,)
        
        # Tensordot or simply matrix mult
        # weights @ modes -> (num_x)
        E_z = np.dot(weights, modes)
        field_evolution[iz, :] = E_z

    intensity_evolution = np.abs(field_evolution)**2

    # --- Calculation of Output Vector ---
    # Projection onto M output ports
    # Output ports positions: W/M * (j + 0.5)
    # Assumed mode shape: Gaussian with same waist as input (roughly)
    output_positions = [W/M * (j + 0.5) for j in range(M)]
    
    # Note: If M != N, beam waist might change? For standard MMI, inputs and outputs are usually similar waveguides.
    # We use beam_waist derived from input geometry heuristic.
    
    output_amplitudes = []
    final_field = field_evolution[-1, :]
    
    for j in range(M):
        center_x_out = output_positions[j]
        # Mode shape for this output
        psi_out = np.exp(-((x_grid - center_x_out)**2) / (beam_waist**2))
        
        # Normalize the output mode to have unit energy 
        # (so that projection gives correct amplitude coefficient)
        norm_factor = np.sqrt(np.sum(np.abs(psi_out)**2) * dx)
        psi_out = psi_out / norm_factor
        
        # Overlap integral: <E_final | psi_out> = int(E_final * conj(psi_out)) dx
        overlap = np.sum(final_field * np.conj(psi_out)) * dx
        output_amplitudes.append(overlap)
        
    output_amplitudes = np.array(output_amplitudes)

    # 6. Visualization & Animation (Optional)
    if output_file is not None:
        fig, (ax_static, ax_anim) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Static Plot: "Top view" intensity map
        extent = [0, L*1e6, 0, W*1e6] # microns
        ax_static.set_title(f"MMI Propagation Field Intensity (L={L*1e6:.1f}um, W={W*1e6:.1f}um)")
        im = ax_static.imshow(intensity_evolution.T, origin='lower', aspect='auto', 
                              extent=extent, cmap='inferno')
        ax_static.set_xlabel("z [um]")
        ax_static.set_ylabel("x [um]")
        
        # Mark input/output positions
        for y_pos in input_positions:
            ax_static.text(0, y_pos*1e6, 'In', color='white', ha='right', va='center', fontsize=8)
            
        for y_pos in output_positions:
            ax_static.text(L*1e6, y_pos*1e6, 'Out', color='white', ha='left', va='center', fontsize=8)

        # Vertical line moving with animation
        line, = ax_static.plot([0, 0], [0, W*1e6], 'w--', lw=1.5)
        
        # Dynamic Plot: Cross-section intensity I(x)
        ax_anim.set_title("Cross-section Intensity Profile")
        ax_anim.set_xlim(0, W*1e6)
        max_intensity = np.max(intensity_evolution)
        ax_anim.set_ylim(0, max_intensity * 1.1)
        ax_anim.set_xlabel("x [um]")
        ax_anim.set_ylabel("Intensity")
        
        profile_line, = ax_anim.plot(x_grid*1e6, intensity_evolution[0, :], 'b-', lw=2)
        
        def update(frame):
            z_val = z_grid[frame]
            
            # Update vertical line on map
            line.set_data([z_val*1e6, z_val*1e6], [0, W*1e6])
            
            # Update profile
            profile_line.set_ydata(intensity_evolution[frame, :])
            ax_anim.set_title(f"Cross-section at z = {z_val*1e6:.1f} um")
            
            return line, profile_line
        
        ani = animation.FuncAnimation(fig, update, frames=num_z_steps, interval=50, blit=True)
        
        if verbose:
            print(f"Saving animation to {output_file}...")
        
        # Check if we can save as mp4 (requires ffmpeg)
        if animation.writers.is_available('ffmpeg'):
            ani.save(output_file, writer='ffmpeg', fps=30)
        else:
            # Fallback to gif using Pillow
            new_output = output_file.replace('.mp4', '.gif')
            print(f"Warning: ffmpeg not found. Saving as GIF to {new_output} instead.")
            ani.save(new_output, writer='pillow', fps=30)
            
        if verbose:
            print("Done!")
        plt.close(fig)

    return output_amplitudes

if __name__ == "__main__":
    # Test Default Nulling MMI (2x2)
    print("Testing MMI simulation with defaults (2x2 Nulling MMI)...")
    out = simulate_mmi(output_file="mmi_nuller_demo.mp4")
    print(f"Output vector: {out}")
    print(f"Output intensities: {np.abs(out)**2}")
