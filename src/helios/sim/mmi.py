
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import os
import tempfile
import shutil
import subprocess
from joblib import Parallel, delayed


def _wrap_phase_radians(phases_rad):
    """Wrap phases to [0, 2π).

    Parameters
    ----------
    phases_rad : array-like
        Phase values in radians.

    Returns
    -------
    np.ndarray
        Wrapped phases in [0, 2π).
    """
    phases = np.asarray(phases_rad, dtype=float)
    return np.mod(phases, 2 * np.pi)


def _calibrate_phases_genetic_like(
    evaluate_metric,
    n_phases,
    beta=0.8,
    initial_step=np.pi / 2,
    epsilon=1e-4,
    initial_phases=None,
    fixed_indices=None,
    max_outer_iterations=200,
    verbose=False,
):
    """Calibrate phase shifters using a genetic-like coordinate descent.

    This is inspired by the "genetic-like" calibration loops used in PHISE and PHOBos.
    Despite the name, it is a deterministic hill-climb / coordinate descent with a
    decaying step size. It is architecture-independent: the only required ingredient
    is a callable that evaluates the metric for a vector of phases.

    Parameters
    ----------
    evaluate_metric : callable
        Function ``evaluate_metric(phases_rad) -> float`` to minimize.
    n_phases : int
        Number of phase shifters.
    beta : float, default=0.8
        Step decay factor. Must satisfy ``0.5 <= beta < 1``.
    initial_step : float, default=π/2
        Initial phase step size [rad].
    epsilon : float, default=1e-4
        Minimum step size [rad]. The loop stops when the step becomes smaller.
    initial_phases : array-like, optional
        Initial phases [rad]. Defaults to all zeros.
    fixed_indices : set[int] | list[int] | None, optional
        Indices to keep fixed (e.g., {0} to remove global phase degeneracy).
    max_outer_iterations : int, default=200
        Safety cap on the number of outer iterations (step decays).
    verbose : bool, default=False
        If True, prints progress.

    Returns
    -------
    dict
        Dictionary containing:
        - ``metric``: metric history (1D)
        - ``phases``: phase history (2D: steps x n_phases)
        - ``best_metric``: best metric encountered
        - ``best_phases``: best phases [rad]
    """
    if not (0.5 <= beta < 1.0):
        raise ValueError("beta must be in the range [0.5, 1[")
    if n_phases <= 0:
        raise ValueError(f"n_phases must be positive, got {n_phases}.")
    if initial_step <= 0:
        raise ValueError(f"initial_step must be > 0, got {initial_step}.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if max_outer_iterations <= 0:
        raise ValueError(f"max_outer_iterations must be > 0, got {max_outer_iterations}.")

    if initial_phases is None:
        phases = np.zeros(n_phases, dtype=float)
    else:
        phases = np.asarray(initial_phases, dtype=float).copy()
        if phases.shape != (n_phases,):
            raise ValueError(f"initial_phases must have shape ({n_phases},), got {phases.shape}.")

    phases = _wrap_phase_radians(phases)

    fixed = set(fixed_indices or [])
    variable_indices = [i for i in range(n_phases) if i not in fixed]

    metric_history = []
    phases_history = []

    best_metric = float("inf")
    best_phases = phases.copy()

    step = float(initial_step)
    outer_it = 0

    while (step > epsilon) and (outer_it < max_outer_iterations):
        if verbose:
            print(f"--- Iteration {outer_it} --- Δφ={step:.3e} rad")

        for i in variable_indices:
            phases_pos = phases.copy()
            phases_neg = phases.copy()
            phases_pos[i] = (phases_pos[i] + step) % (2 * np.pi)
            phases_neg[i] = (phases_neg[i] - step) % (2 * np.pi)

            m_old = float(evaluate_metric(phases))
            m_pos = float(evaluate_metric(phases_pos))
            m_neg = float(evaluate_metric(phases_neg))

            metric_history.append(m_old)
            phases_history.append(phases.copy())

            if m_old < best_metric:
                best_metric = m_old
                best_phases = phases.copy()
            if m_pos < best_metric:
                best_metric = m_pos
                best_phases = phases_pos.copy()
            if m_neg < best_metric:
                best_metric = m_neg
                best_phases = phases_neg.copy()

            if verbose:
                print(f"Phase {i}: {m_neg:.3e} | {m_old:.3e} | {m_pos:.3e}")

            # Minimize metric (pick best of {neg, old, pos})
            if (m_pos < m_old) and (m_pos < m_neg):
                phases = phases_pos
            elif (m_neg < m_old) and (m_neg < m_pos):
                phases = phases_neg

            # Enforce fixed indices exactly (avoid drift due to numerical ops)
            for j in fixed:
                phases[j] = float(_wrap_phase_radians(phases[j]))

        step *= beta
        outer_it += 1

    # Record final state
    metric_history.append(float(evaluate_metric(phases)))
    phases_history.append(phases.copy())
    if metric_history[-1] < best_metric:
        best_metric = metric_history[-1]
        best_phases = phases.copy()

    return {
        "metric": np.asarray(metric_history, dtype=float),
        "phases": np.asarray(phases_history, dtype=float),
        "best_metric": float(best_metric),
        "best_phases": _wrap_phase_radians(best_phases),
    }


def calibrate_input_phases_genetic(
    N=4,
    M=4,
    L=None,
    W=10.0e-6,
    n_eff=2.0458,
    wavelength=1.55e-6,
    input_amplitudes=None,
    bright_output_idx=0,
    num_modes=50,
    num_z_steps=None,
    z_resolution=None,
    Din=None,
    Dout=None,
    beta=0.8,
    initial_step=np.pi / 2,
    epsilon=1e-4,
    verbose=False,
):
    """Calibrate input phases to redirect flux to a bright output.

    The phase shifters are modeled as the input phases (one per input). The objective is
    to minimize the null-depth-like metric:

    ``metric = sum(null_outputs) / bright_output``

    where "null outputs" are all outputs except the chosen bright output.

        Notes
        -----
        - The global phase is physically irrelevant when the metric depends only on output
            intensities (as it does here). Therefore the solution is not unique: adding a
            constant phase offset to all inputs yields the same metric.
        - We intentionally do *not* fix the phase of any input as a reference. This keeps
            the input-1 phase as a free degree of freedom, which can be important when
            interfacing with hardware that has no absolute phase origin.
        - This uses a genetic-like coordinate-descent algorithm with decaying step, inspired
            by PHISE/PHOBos calibration utilities.

    Parameters
    ----------
    N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution :
        Same meaning as in :func:`simulate`.
    bright_output_idx : int, default=0
        Output index to maximize (the "Bright" output).
    Din, Dout : float, optional
        Input/output port spacing [m]. See :func:`simulate`.
    beta, initial_step, epsilon : float
        Calibration loop parameters.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Dictionary with:
        - ``metric``: metric history
        - ``phases``: phase history [rad]
        - ``best_metric``: best metric
        - ``best_phases``: best phases [rad]
        - ``bright_output_idx``: bright output index
    """
    if not (0 <= bright_output_idx < M):
        raise ValueError(f"bright_output_idx must be in [0, {M-1}], got {bright_output_idx}.")

    # Use the same default L heuristic as simulate().
    if L is None:
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2

    if input_amplitudes is None:
        input_amplitudes = [1.0 / np.sqrt(N)] * N
    if len(input_amplitudes) != N:
        raise ValueError(f"Length of input_amplitudes ({len(input_amplitudes)}) must match N ({N})")

    input_amplitudes = np.asarray(input_amplitudes, dtype=complex)
    magnitudes = np.abs(input_amplitudes)
    start_phases = _wrap_phase_radians(np.angle(input_amplitudes))

    def evaluate_metric(phases_rad):
        phases_rad = _wrap_phase_radians(phases_rad)
        amps = magnitudes * np.exp(1j * phases_rad)

        out = simulate(
            N=N,
            M=M,
            L=L,
            W=W,
            n_eff=n_eff,
            wavelength=wavelength,
            input_amplitudes=amps,
            num_modes=num_modes,
            num_z_steps=num_z_steps,
            z_resolution=z_resolution,
            output_file=None,
            verbose=False,
            Din=Din,
            Dout=Dout,
        )

        intensities = np.abs(out) ** 2
        bright = float(intensities[bright_output_idx])
        null_sum = float(np.sum(intensities) - bright)

        if bright <= 0:
            return float("inf")
        return null_sum / bright

    result = _calibrate_phases_genetic_like(
        evaluate_metric=evaluate_metric,
        n_phases=N,
        beta=beta,
        initial_step=initial_step,
        epsilon=epsilon,
        initial_phases=start_phases,
        fixed_indices=None,
        verbose=verbose,
    )
    result["bright_output_idx"] = int(bright_output_idx)
    return result


def _compute_symmetric_port_positions(num_ports, W, spacing, name):
    """Compute symmetric port positions around x=W/2.

    Parameters
    ----------
    num_ports : int
        Number of ports.
    W : float
        MMI width [m].
    spacing : float | None
        Port-to-port spacing [m]. If None, uses the historical default spacing W/num_ports.
    name : str
        Human-readable name used for error messages (e.g., "input", "output").

    Returns
    -------
    list[float]
        Port center positions along x in [m], symmetric about x=W/2.

    Raises
    ------
    ValueError
        If spacing is non-positive or causes ports to lie outside the MMI [0, W].
    """
    if num_ports <= 0:
        raise ValueError(f"{name} ports must be a positive integer, got {num_ports}.")
    if W <= 0:
        raise ValueError(f"MMI width W must be positive, got {W}.")

    if spacing is None:
        spacing = W / num_ports
    if spacing <= 0:
        raise ValueError(f"{name} spacing must be > 0, got {spacing}.")

    center = 0.5 * W
    offsets = (np.arange(num_ports, dtype=float) - 0.5 * (num_ports - 1)) * spacing
    positions = center + offsets

    # Numerical tolerance in meters (scaled with W).
    eps = max(1e-15, 1e-15 * abs(W))
    min_pos = float(np.min(positions))
    max_pos = float(np.max(positions))
    if (min_pos < -eps) or (max_pos > W + eps):
        raise ValueError(
            f"{name} spacing {spacing} m is too large for W={W} m: "
            f"{name} positions would span [{min_pos}, {max_pos}] m outside [0, {W}] m."
        )

    # Clamp tiny numerical noise at boundaries.
    positions = np.clip(positions, 0.0, W)
    return positions.tolist()

def _compute_mmi_field(N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose=False, Din=None, Dout=None):
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

    input_positions = _compute_symmetric_port_positions(N, W, Din, name="input")
    
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

    output_positions = _compute_symmetric_port_positions(M, W, Dout, name="output")
    
    return z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx

# --- Parallel Rendering Helpers ---

def _render_frame_static(frame_idx, z_grid, x_grid, intensity_evolution, L, W, input_positions, output_positions, output_dir):
    """Render a single frame for simulate."""
    # Ensure Agg backend for thread safety / headless
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    
    z_val = z_grid[frame_idx]
    
    fig, (ax_static, ax_anim) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 1. Static Plot
    extent = [0, L*1e6, 0, W*1e6] # microns
    ax_static.set_title(f"MMI Propagation Field Intensity (L={L*1e6:.1f}um, W={W*1e6:.1f}um)")
    ax_static.imshow(intensity_evolution.T, origin='lower', aspect='auto', 
                          extent=extent, cmap='inferno')
    ax_static.set_xlabel("z [um]")
    ax_static.set_ylabel("x [um]")
    
    # Inputs/Outputs markers
    for y_pos in input_positions:
        ax_static.text(0, y_pos*1e6, 'In', color='white', ha='right', va='center', fontsize=8)
    for y_pos in output_positions:
        ax_static.text(L*1e6, y_pos*1e6, 'Out', color='white', ha='left', va='center', fontsize=8)

    # Moving vertical line
    ax_static.plot([z_val*1e6, z_val*1e6], [0, W*1e6], 'w--', lw=1.5)
    
    # 2. Dynamic Plot
    ax_anim.set_title(f"Cross-section at z = {z_val*1e6:.1f} um")
    ax_anim.set_xlim(0, W*1e6)
    max_intensity = np.max(intensity_evolution)
    ax_anim.set_ylim(0, max_intensity * 1.1)
    ax_anim.set_xlabel("x [um]")
    ax_anim.set_ylabel("Intensity")
    
    ax_anim.plot(x_grid*1e6, intensity_evolution[frame_idx, :], 'b-', lw=2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close(fig)

def _render_contrib_frame_static(frame_idx, z_grid, x_grid, intensity_total_evol, phasors, L, W, input_positions, output_positions, N, M, output_dir):
    """Render a single frame for simulate_contributions."""
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    z_val = z_grid[frame_idx]

    fig = plt.figure(figsize=(10, 16))
    gs = fig.add_gridspec(4, M, height_ratios=[1.5, 1, 1, 1])
    
    # MMI Plots (Top rows)
    ax_static = fig.add_subplot(gs[0, :])
    ax_anim = fig.add_subplot(gs[1, :])
    
    # Polar Plots (3rd Row)
    polar_axes = []
    for j in range(M):
        ax_p = fig.add_subplot(gs[2, j], projection='polar')
        ax_p.set_title(f"Output {j+1}", fontsize=10)
        polar_axes.append(ax_p)

    # Z-Profile Plots (4th Row) - Grouped
    z_axes = []
    # Single subplot spanning all columns
    ax_z = fig.add_subplot(gs[3, :])
    ax_z.set_title(f"Z-Profile All Outputs", fontsize=9)
    ax_z.set_xlabel("z [um]")
    z_axes.append(ax_z)
        
    # -- Static Plot --
    extent = [0, L*1e6, 0, W*1e6]
    ax_static.set_title(f"Field Intensity (L={L*1e6:.1f}um)")
    ax_static.imshow(intensity_total_evol.T, origin='lower', aspect='auto', extent=extent, cmap='inferno')
    ax_static.set_xlabel("z [um]")
    ax_static.set_ylabel("x [um]")
    ax_static.plot([z_val*1e6, z_val*1e6], [0, W*1e6], 'w--', lw=1.5)
    
    # Markers
    ax_static.scatter([0]*N, [p*1e6 for p in input_positions], color='white', marker='o', s=20, zorder=10)
    ax_static.scatter([L*1e6]*M, [p*1e6 for p in output_positions], color='white', marker='o', s=20, zorder=10)

    # -- Profile Plot --
    ax_anim.set_title(f"Cross-section at z={z_val*1e6:.1f} um")
    ax_anim.set_xlim(0, W*1e6)
    ax_anim.set_ylim(0, np.max(intensity_total_evol)*1.1)
    ax_anim.set_xlabel("x [um]")
    ax_anim.plot(x_grid*1e6, intensity_total_evol[frame_idx, :], 'b-', lw=2)
    
    for pos in output_positions:
        ax_anim.axvline(x=pos*1e6, color='k', linestyle=':', linewidth=0.8, alpha=0.7)
    
    # -- Polar Plots --
    colors = plt.cm.get_cmap('hsv', N+1)
    max_coupling = np.max(np.abs(phasors)) 
    
    for j in range(M):
        ax_p = polar_axes[j]
        # Fixed scale for stability
        ax_p.set_ylim(0, 1.1 * max_coupling if max_coupling > 1e-9 else 1.0)
        
        # Individual phasors
        for i in range(N):
            val = phasors[frame_idx, j, i]
            ax_p.plot([0, np.angle(val)], [0, np.abs(val)], color=colors(i), lw=2, label=f"In {i+1}" if frame_idx==0 else "")
        
        # Total phasor
        tot = np.sum(phasors[frame_idx, j, :])
        ax_p.plot([0, np.angle(tot)], [0, np.abs(tot)], 'k--', lw=2, label="Total" if frame_idx==0 else "")
        

        if j == M-1:
            # Legend (simplified)
            # ax_p.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
            pass

    # -- Z-Profile Plots (Grouped) --
    ax_z = z_axes[0] # Single axis
    
    # Max intensity for scaling
    max_int_z = np.max(intensity_total_evol)*1.1
    z_colors = plt.cm.get_cmap('tab10', M)

    for j in range(M):
        # Find x index for this output
        x_out = output_positions[j]
        ix = np.argmin(np.abs(x_grid - x_out))
        
        # Extract I(z) at this x
        I_z = intensity_total_evol[:, ix]
        
        ax_z.plot(z_grid*1e6, I_z, color=z_colors(j), lw=1.5, label=f'Out {j+1}' if frame_idx==0 else "")
        
        # Moving Point
        ax_z.plot(z_val*1e6, I_z[frame_idx], 'o', color=z_colors(j), markersize=4)
        
    # Moving Vertical Line
    ax_z.axvline(x=z_val*1e6, color='r', linestyle='--', lw=1.5)
        
    ax_z.set_xlim(0, L*1e6)
    ax_z.set_ylim(0, max_int_z)
    if frame_idx == 0:
        ax_z.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close(fig)

def _make_video_from_frames(output_file, frame_dir, fps=30):
    """Stitch frames into video using ffmpeg."""
    # Check for ffmpeg
    if shutil.which('ffmpeg') is None:
        print("Error: ffmpeg not found. Cannot generate video.")
        return

    # ffmpeg command
    # -y to overwrite
    # -i frame_%05d.png
    # -c:v libx264 -pix_fmt yuv420p
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _compute_single_field_wrapper(i, N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, Din, Dout):
    """Wrapper to compute field for a single input (parallel helper)."""
    single_input = np.zeros(N, dtype=complex)
    single_input[i] = input_amplitudes[i]
    
    # We only need the field_evolution (3rd return, index 2)
    ret = _compute_mmi_field(
        N, M, L, W, n_eff, wavelength, single_input, num_modes, num_z_steps, z_resolution, verbose=False, Din=Din, Dout=Dout
    )
    return ret[2]

def simulate(N=2, M=2, L=None, W=10.0e-6, n_eff=2.0458, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=None, z_resolution=None, output_file=None, verbose=False, Din=None, Dout=None):


    """
    Simulates light propagation in an NxM MMI (Multi-Mode Interferometer) using Eigenmode Expansion (Hard Wall).
    
    This function models the propagation of light through a multimode waveguide section, calculating
    the output amplitudes at specified output ports. It uses a hard-wall approximation for the
    guided modes and assumes a step-index profile.

    Parameters
    ----------
    N : int, default=2
        Number of input ports.
    M : int, default=2
        Number of output ports.
    L : float, optional
        Length of the MMI region [m]. If None, it is automatically calculated for 
        paired interference at the specified width and wavelength.
    W : float, default=10.0e-6
        Width of the MMI region [m].
    n_eff : float, default=2.0458
        Effective refractive index of the MMI slab.
    wavelength : float, default=1.55e-6
        Operating wavelength [m].
    input_amplitudes : list or array, optional
        Complex amplitudes for the N inputs. If None, defaults to uniform illumination
        normalized by 1/sqrt(N).
    num_modes : int, default=50
        Number of eigenmodes to use for the decomposition. Higher numbers increase accuracy
        but also computation time.
    num_z_steps : int, optional
        Number of steps for z-propagation grid. If None, calculated from `z_resolution`.
    z_resolution : float, optional
        Step size in z [m]. Defaults to wavelength/30 if `num_z_steps` is also None.
    Din : float, optional
        Input port spacing [m]. If None, uses the historical default spacing ``W/N``.
        Inputs are placed symmetrically about the centerline at x = W/2.
    Dout : float, optional
        Output port spacing [m]. If None, uses the historical default spacing ``W/M``.
        Outputs are placed symmetrically about the centerline at x = W/2.
    output_file : str, optional
        Path to save an animation of the propagation (e.g., 'mmi_prop.mp4'). 
        If None, no animation is generated.
    verbose : bool, default=False
        If True, prints detailed simulation progress and parameters.

    Returns
    -------
    np.ndarray
        Complex amplitudes at the M output ports.
    """
    
    # Defaults logic        
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
        N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din=Din, Dout=Dout
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
    # 6. Visualization & Animation (Optional)
    if output_file is not None:
        if verbose:
            print(f"Generating animation frames in parallel for {output_file}...")
        
        # Temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Parallel Rendering
            num_cores = -1 # Use all cores
            Parallel(n_jobs=num_cores)(
                delayed(_render_frame_static)(
                    idx, z_grid, x_grid, intensity_evolution, L, W, input_positions, output_positions, temp_dir
                ) for idx in tqdm(range(num_z_steps), desc="Rendering Frames", disable=not verbose)
            )
            
            if verbose:
                print("Stitching frames with ffmpeg...")
            
            _make_video_from_frames(output_file, temp_dir, fps=30)
            
        if verbose:
            print("Done!")

    return output_amplitudes

def compute_contributions(N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose=False, Din=None, Dout=None):
    """
    Calculates MMI fields and contributions, returning raw data for analysis or custom plotting.

    This function performs the full EME simulation including individual simulations for each input
    to determine the complex coupling coefficients (phasors) from each input to each output.

    Parameters
    ----------
    N : int
        Number of input ports.
    M : int
        Number of output ports.
    L : float, optional
        Length of the MMI region [m].
    W : float
        Width of the MMI region [m].
    n_eff : float
        Effective refractive index.
    wavelength : float
        Wavelength [m].
    input_amplitudes : list
        Input complex amplitudes.
    num_modes : int
        Number of modes for EME.
    num_z_steps : int, optional
        Number of Z steps.
    z_resolution : float, optional
        Z resolution [m].
    verbose : bool, default=False
        Print status.
    Din : float, optional
        Input port spacing [m]. If None, uses the historical default spacing ``W/N``.
        Inputs are placed symmetrically about x = W/2.
    Dout : float, optional
        Output port spacing [m]. If None, uses the historical default spacing ``W/M``.
        Outputs are placed symmetrically about x = W/2.

    Returns
    -------
    dict
        A dictionary containing simulation data:
        
        - ``z_grid``: Array of z positions.
        - ``x_grid``: Array of x positions.
        - ``intensity_total_evol``: 2D array of total field intensity (z, x).
        - ``phasors``: Complex array of shape (num_z, M, N) representing the coupling from each input i to output j at each z step.
        - ``input_positions``: List of input port x-positions.
        - ``output_positions``: List of output port x-positions.
        - ``L``, ``W``, ``N``, ``M``: Geometry parameters.
        - ``num_z_steps``: Actual number of z steps used.
    """
    # Defaults logic
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
        N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din=Din, Dout=Dout
    )
    
    # Update num_z_steps to match (important for animation frames)
    num_z_steps = len(z_grid)
    
    # Store total intensity for main plot
    intensity_total_evol = np.abs(field_total_evol)**2

    # Now compute individual contributions
    # For each input i, simulate with only input_amplitudes[i] active
    if verbose:
        print("Computing separate field contributions (Parallel)...")
    
    contributions_fields = Parallel(n_jobs=-1)(
        delayed(_compute_single_field_wrapper)(
            i, N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, Din, Dout
        ) for i in range(N)
    )
        
    # Pre-compute Overlaps (Phasors) for all z steps
    phasors = np.zeros((num_z_steps, M, N), dtype=complex)
    
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

    return {
        "z_grid": z_grid,
        "x_grid": x_grid,
        "intensity_total_evol": intensity_total_evol,
        "phasors": phasors,
        "input_positions": input_positions,
        "output_positions": output_positions,
        "L": L,
        "W": W,
        "N": N,
        "M": M,
        "num_z_steps": num_z_steps
    }

def simulate_contributions(N=2, M=2, L=None, W=10.0e-6 , n_eff=2.0458, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=None, z_resolution=None, output_file=None, verbose=False, Din=None, Dout=None):
    """
    Simulates light propagation with explicit visualization of phasor contributions from each input.
    
    This wrapper calls ``compute_contributions`` and then optionally generates a detailed video
    showing the total field evolution alongside polar plots of the complex contributions from each 
    input to the outputs.
    
    Parameters
    ----------
    N : int, default=2
        Number of input ports.
    M : int, default=2
        Number of output ports.
    L : float, optional
        Length of the MMI [m].
    W : float, default=10.0e-6
        Width of the MMI [m].
    n_eff : float, default=2.0458
        Effective refractive index.
    wavelength : float, default=1.55e-6
        Wavelength [m].
    input_amplitudes : list, optional
        Input complex amplitudes.
    num_modes : int, default=50
        Number of modes.
    num_z_steps : int, optional
        Number of z steps.
    z_resolution : float, optional
        Z resolution.
    output_file : str, optional
        If provided, generates a detailed MP4 animation.
    verbose : bool, default=False
        Print status.
    Din : float, optional
        Input port spacing [m]. If None, uses the historical default spacing ``W/N``.
        Inputs are placed symmetrically about x = W/2.
    Dout : float, optional
        Output port spacing [m]. If None, uses the historical default spacing ``W/M``.
        Outputs are placed symmetrically about x = W/2.

    Returns
    -------
    np.ndarray
        Complex amplitudes at the M outputs.
    """
    # 1. Calculate Data
    data = compute_contributions(N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din=Din, Dout=Dout)
    
    z_grid = data["z_grid"]
    x_grid = data["x_grid"]
    intensity_total_evol = data["intensity_total_evol"]
    phasors = data["phasors"]
    input_positions = data["input_positions"]
    output_positions = data["output_positions"]
    L = data["L"]
    W = data["W"] # Ensure W is retrieved if defaulted inside
    num_z_steps = data["num_z_steps"]
    
    # 2. Visualization
    if output_file is not None:
        if verbose:
            print(f"Generating contributions animation frames in parallel for {output_file}...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            Parallel(n_jobs=-1)(
                delayed(_render_contrib_frame_static)(
                    idx, z_grid, x_grid, intensity_total_evol, phasors, L, W, input_positions, output_positions, N, M, temp_dir
                ) for idx in tqdm(range(num_z_steps), desc="Rendering Frames", disable=not verbose)
            )
            
            if verbose:
                 print("Stitching frames with ffmpeg...")
                 
            _make_video_from_frames(output_file, temp_dir, fps=30)
        
    # Return total output
    output_amplitudes = simulate(N, M, L, W, n_eff, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, output_file=None, verbose=verbose, Din=Din, Dout=Dout)

    if verbose:
        print(f"Output amplitudes: {output_amplitudes}")

    return output_amplitudes
