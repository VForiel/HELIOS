
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

def _compute_mmi_field(N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose=False):
    """
    Core field calculation (Internal helper).
    Returns:
        z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx
    """
    k0 = 2 * np.pi / wavelength
    
    # Defaults for z-resolution if num_z_steps is not explicit
    if num_z_steps is None:
        if z_resolution is None:
            # Default to wavelength / 30.0 (1 sec video @ 30fps = 1 wavelength)
            z_resolution = wavelength / 30.0
            if verbose:
                print(f"Using default z-resolution: lambda/30 = {z_resolution*1e6:.3f} um")
    
        # Calculate num_z_steps based on resolution
        # num_steps = ceil(L / res) + 1 to cover 0 to L
        num_z_steps = int(np.ceil(L / z_resolution)) + 1
        if verbose:
            print(f"Calculated num_z_steps = {num_z_steps} for L = {L*1e6:.1f} um")
    elif verbose and z_resolution is not None:
        print(f"Warning: num_z_steps ({num_z_steps}) provided, ignoring z_resolution ({z_resolution})")
    
    # ... (Geometry setup remains same, skipped in diff if unchanged nearby)
    pass 
    # Logic continues... but I need to inject progress bar in the propagation loop (step 5)
    # I cannot skip lines in replace_file_content easily without matching.
    # So I will just rewrite the surrounding lines and replace _compute_mmi_field partly? 
    # Or just replace the imports and the loop.
    
    # Let's do imports first.

    input_positions = [W/N * (i + 0.5) for i in range(N)]
    
    # 2. Define Waveguide Modes (Hard Wall Approximation)
    x_grid = np.linspace(0, W, 500)
    dx = x_grid[1] - x_grid[0]
    
    modes = []
    betas = []
    
    for m in range(num_modes):
        kx_m = (m + 1) * np.pi / W
        sq_term = (k0 * n_eff)**2 - kx_m**2
        if sq_term < 0:
            beta_m = 0 
        else:
            beta_m = np.sqrt(sq_term)
            
        betas.append(beta_m)
        phi_m = np.sqrt(2/W) * np.sin(kx_m * x_grid)
        modes.append(phi_m)
        
    betas = np.array(betas)
    modes = np.array(modes) # Shape: (num_modes, num_x_points)
    
    # 3. Construct Input Field (Gaussian beams)
    input_field = np.zeros_like(x_grid, dtype=complex)
    beam_waist = (W / N) / 4 
    
    if verbose:
        print(f"Injecting input vector: {input_amplitudes}")
    
    for idx, amp in enumerate(input_amplitudes):
        if amp == 0:
            continue
        center_x = input_positions[idx]
        gauss = np.exp(-((x_grid - center_x)**2) / (beam_waist**2))
        norm_factor = np.sqrt(np.sum(np.abs(gauss)**2) * dx)
        gauss = gauss / norm_factor
        input_field += amp * gauss

    # 4. Mode Decomposition
    coeffs = []
    for m in range(num_modes):
        c_m = np.sum(input_field * modes[m]) * dx
        coeffs.append(c_m)
    coeffs = np.array(coeffs)
    
    # 5. Propagation
    z_grid = np.linspace(0, L, num_z_steps)
    field_evolution = np.zeros((num_z_steps, len(x_grid)), dtype=complex)
    
    iterator = enumerate(z_grid)
    if verbose:
        # Use tqdm for the loop
        iterator = enumerate(tqdm(z_grid, desc="Simulating Propagation", unit="step"))
        
    for iz, z in iterator:
        phase_term = np.exp(-1j * betas * z)
        weights = coeffs * phase_term
        E_z = np.dot(weights, modes)
        field_evolution[iz, :] = E_z

    output_positions = [W/M * (j + 0.5) for j in range(M)]
    
    return z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx


def simulate_mmi(N=2, M=2, L=None, W=None, n_eff=2.0458, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=None, z_resolution=None, output_file=None, verbose=False):
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
        num_z_steps (int, optional): Number of steps for z-propagation. If None, calculated from z_resolution.
        z_resolution (float, optional): Step size in z [m]. Defaults to wavelength/10 if num_z_steps is also None.
        output_file (str, optional): Filename for the output animation. If None, no animation is generated.
        verbose (bool): If True, prints detailed simulation steps. Defaults to False.

    Returns:
        np.array: Complex amplitudes at the M outputs.
    """
    
    # Defaults logic
    if W is None:
        W = 10.0e-6 # 10 um default width
        
    if L is None:
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2
        if verbose:
            print(f"Auto-calculated L = {L*1e6:.2f} um for W = {W*1e6:.2f} um (Paired Interference)")
            
    # Default for num_z_steps handled in _compute_mmi_field if not provided here.
    # However, for the animation code which uses num_z_steps (specifically frames=num_z_steps),
    # we need the computed value back from _compute_mmi_field.
    # Fortunately _compute_mmi_field returns z_grid, so we can re-derive len(z_grid).

    if input_amplitudes is None:
        val = 1.0 / np.sqrt(N)
        input_amplitudes = [val] * N
    
    if len(input_amplitudes) != N:
        raise ValueError(f"Length of input_amplitudes ({len(input_amplitudes)}) must match N ({N})")

    # Run internal simulation
    z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx = _compute_mmi_field(
        N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose
    )
    
    # Update num_z_steps to match the actual grid size used
    num_z_steps = len(z_grid)

    intensity_evolution = np.abs(field_evolution)**2

    # --- Calculation of Output Vector ---
    output_amplitudes = []
    final_field = field_evolution[-1, :]
    
    for j in range(M):
        center_x_out = output_positions[j]
        # Mode shape for this output
        psi_out = np.exp(-((x_grid - center_x_out)**2) / (beam_waist**2))
        
        # Normalize the output mode to have unit energy 
        norm_factor = np.sqrt(np.sum(np.abs(psi_out)**2) * dx)
        psi_out = psi_out / norm_factor
        
        # Overlap integral
        overlap = np.sum(final_field * np.conj(psi_out)) * dx
        output_amplitudes.append(overlap)
        
    output_amplitudes = np.array(output_amplitudes)

    if verbose:
        print(f"Output amplitudes: {output_amplitudes}")

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
        
        # Animation Progress Bar
        if verbose:
            pbar = tqdm(total=num_z_steps, desc="Saving Frames", unit="frame")
            
        def progress_callback(current, total):
            if verbose and pbar:
                pbar.update(current - pbar.n)
                if current == total:
                    pbar.close()
        
        # Check if we can save as mp4 (requires ffmpeg)
        if animation.writers.is_available('ffmpeg'):
            ani.save(output_file, writer='ffmpeg', fps=30, progress_callback=progress_callback)
        else:
            # Fallback to gif using Pillow
            new_output = output_file.replace('.mp4', '.gif')
            print(f"Warning: ffmpeg not found. Saving as GIF to {new_output} instead.")
            ani.save(new_output, writer='pillow', fps=30, progress_callback=progress_callback)
            
        if verbose:
            print("Done!")
        plt.close(fig)

    return output_amplitudes

def simulate_mmi_contributions(N=2, M=2, L=None, W=None, n_eff=2.0458, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=None, z_resolution=None, output_file=None, verbose=False):
    """
    Simulates light propagation with phasor contributions from each input.
    """
    # Defaults logic
    if W is None:
        W = 10.0e-6 
    if L is None:
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2
        if verbose:
            print(f"Auto-calculated L = {L*1e6:.2f} um")
    if input_amplitudes is None:
        val = 1.0 / np.sqrt(N)
        input_amplitudes = [val] * N
    if len(input_amplitudes) != N:
        raise ValueError(f"Length mismatch")

    # 1. Individual Simulations
    contributions_fields = [] # List of N arrays (num_z, num_x)
    
    # Common geometry data from first run
    z_grid, x_grid, field_total_evol, output_positions, input_positions, beam_waist, dx = _compute_mmi_field(
        N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose
    )
    
    # Update num_z_steps to match (important for animation frames)
    num_z_steps = len(z_grid)
    
    # Store total intensity for main plot
    intensity_total_evol = np.abs(field_total_evol)**2

    # Now compute individual contributions
    # For each input i, simulate with only input_amplitudes[i] active
    for i in range(N):
        single_input = np.zeros(N, dtype=complex)
        single_input[i] = input_amplitudes[i]
        
        # Reuse same grid steps
        _, _, field_evol_i, _, _, _, _ = _compute_mmi_field(
            N, M, L, W, n_eff, wavelength, single_input, num_modes, num_z_steps, z_resolution, verbose=False
        )
        contributions_fields.append(field_evol_i)
        
    # Pre-compute Overlaps (Phasors) for all z steps
    # phasors[z_idx][output_j][input_i] -> complex number
    phasors = np.zeros((num_z_steps, M, N), dtype=complex)
    
    # Pre-compute output mode shapes centered at output_positions
    # We assume output guides are straight?? Or simply we project onto the mode at THAT z?
    # Usually we project onto the mode *at the output plane*.
    # Valid question: "contribution... à l'instant (ou position z) donné".
    # This implies projecting onto a local mode. 
    # But mode overlap is only meaningful if a waveguide exists there.
    # We will project onto the *Output Mode Shape* shifted to the output position, 
    # effectively asking "How much of Input I is in the mode that leads to Output J *at* Z?"
    
    output_modes = [] # shape (M, num_x)
    for j in range(M):
        center_x_out = output_positions[j]
        psi = np.exp(-((x_grid - center_x_out)**2) / (beam_waist**2))
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        output_modes.append(psi / norm)
        
    for iz in range(num_z_steps):
        for j in range(M): # Output J
            psi_out = output_modes[j]
            for i in range(N): # Input I
                E_i_z = contributions_fields[i][iz, :]
                # Overlap
                coupling = np.sum(E_i_z * np.conj(psi_out)) * dx
                phasors[iz, j, i] = coupling

    # 2. Visualization
    if output_file is not None:
        # Layout: Top = MMI (static), Middle = Profile, Bottom = M Polar Plots
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, M, height_ratios=[1, 1, 1])
        
        # MMI Plots (Top rows)
        ax_static = fig.add_subplot(gs[0, :])
        ax_anim = fig.add_subplot(gs[1, :])
        
        # Polar Plots (Bottom Row)
        polar_axes = []
        for j in range(M):
            ax_p = fig.add_subplot(gs[2, j], projection='polar')
            ax_p.set_title(f"Output {j+1}", fontsize=10)
            polar_axes.append(ax_p)
            
        # -- Setup Static Plot --
        extent = [0, L*1e6, 0, W*1e6]
        ax_static.set_title(f"Field Intensity (L={L*1e6:.1f}um)")
        ax_static.imshow(intensity_total_evol.T, origin='lower', aspect='auto', extent=extent, cmap='inferno')
        ax_static.set_xlabel("z [um]")
        ax_static.set_ylabel("x [um]")
        line, = ax_static.plot([0, 0], [0, W*1e6], 'w--', lw=1.5)
        
        # Add points at input and output positions
        # Input positions at z=0
        ax_static.scatter([0]*N, [p*1e6 for p in input_positions], color='white', marker='o', s=20, zorder=10)
        # Output positions at z=L
        ax_static.scatter([L*1e6]*M, [p*1e6 for p in output_positions], color='white', marker='o', s=20, zorder=10)

        # -- Setup Profile Plot --
        ax_anim.set_title("Cross-section Intensity")
        ax_anim.set_xlim(0, W*1e6)
        ax_anim.set_ylim(0, np.max(intensity_total_evol)*1.1)
        ax_anim.set_xlabel("x [um]")
        profile_line, = ax_anim.plot(x_grid*1e6, intensity_total_evol[0, :], 'b-', lw=2)
        
        # Add fine dotted lines at output positions
        for pos in output_positions:
            ax_anim.axvline(x=pos*1e6, color='k', linestyle=':', linewidth=0.8, alpha=0.7)
        
        # -- Setup Polar Plots --
        # For each output j, we have N arrows + 1 Sum arrow
        # We store the quiver/plot objects
        arrows_lists = [] # list of M lists of (N) arrows
        sum_arrows = []   # list of M sum arrows
        
        colors = plt.cm.get_cmap('hsv', N+1) # Input colors
        
        for j in range(M):
            ax_p = polar_axes[j]
            ax_p.set_ylim(0, 1.1 * np.max(np.abs(phasors))) # Scale to max coupling
            
            p_arrows = []
            for i in range(N):
                # Initial arrow (z=0)
                val = phasors[0, j, i]
                arr = ax_p.plot([0, np.angle(val)], [0, np.abs(val)], color=colors(i), lw=2, label=f"In {i+1}")[0]
                p_arrows.append(arr)
            
            # Sum arrow (Total)
            tot = np.sum(phasors[0, j, :])
            sum_arr = ax_p.plot([0, np.angle(tot)], [0, np.abs(tot)], 'k--', lw=2, label="Total")[0]
            
            arrows_lists.append(p_arrows)
            sum_arrows.append(sum_arr)
            
            # Legend only on first polar plot to save space
            if j == M-1:
                # Legend placement might need adjustment for the new layout
                ax_p.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

        plt.tight_layout()

        def update(frame):
            z_val = z_grid[frame]
            
            # 1. Update Line
            line.set_data([z_val*1e6, z_val*1e6], [0, W*1e6])
            
            # 2. Update Profile
            profile_line.set_ydata(intensity_total_evol[frame, :])
            ax_anim.set_title(f"Cross-section at z={z_val*1e6:.1f} um")
            
            # 3. Update Polar Plots
            for j in range(M):
                # Update individual input phasors
                for i in range(N):
                    val = phasors[frame, j, i]
                    # Update plot data (theta, r) pair
                    # Matplotlib polar plot requires [theta_start, theta_end], [r_start, r_end]
                    arrows_lists[j][i].set_data([0, np.angle(val)], [0, np.abs(val)])
                
                # Update total phasor
                tot = np.sum(phasors[frame, j, :])
                sum_arrows[j].set_data([0, np.angle(tot)], [0, np.abs(tot)])
                
            return [line, profile_line] + [a for al in arrows_lists for a in al] + sum_arrows

        ani = animation.FuncAnimation(fig, update, frames=num_z_steps, interval=50, blit=False) # blit=False for polar often safer
        
        if verbose:
            print(f"Saving contributions animation to {output_file}...")

        # Animation Progress Bar for contributions
        if verbose:
            pbar_contrib = tqdm(total=num_z_steps, desc="Saving Frames", unit="frame")

        def progress_callback_contrib(current, total):
            if verbose and pbar_contrib:
                pbar_contrib.update(current - pbar_contrib.n)
                if current == total:
                    pbar_contrib.close()
            
        if animation.writers.is_available('ffmpeg'):
            ani.save(output_file, writer='ffmpeg', fps=30, progress_callback=progress_callback_contrib)
        else:
            new_output = output_file.replace('.mp4', '.gif')
            print(f"Warning: ffmpeg not found. Saving as GIF to {new_output} instead.")
            ani.save(new_output, writer='pillow', fps=30, progress_callback=progress_callback_contrib)
            
        plt.close(fig)
        
    # Return total output
    output_amplitudes = simulate_mmi(N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, output_file=None, verbose=verbose)

    if verbose:
        print(f"Output amplitudes: {output_amplitudes}")

    return output_amplitudes
