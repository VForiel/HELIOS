
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

# Import LP mode calculation utilities
try:
    from .lp_modes import (
        compute_v_number,
        compute_multimode_coupling,
        print_mode_info,
    )
    _HAS_LP_MODES = True
except ImportError:
    _HAS_LP_MODES = False
    print("⚠️ LP modes module not available - using Gaussian approximation only")


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
    n_core=2.0458,
    delta_n=0.0958,
    wavelength=1.55e-6,
    input_amplitudes=None,
    bright_output_idx=0,
    num_modes=50,
    num_z_steps=None,
    z_resolution=None,
    Din=None,
    Dout=None,
    Sin=None,
    Sout=None,
    beta=0.8,
    initial_step=np.pi / 2,
    epsilon=1e-4,
    verbose=False,
):
    """Calibrate input phases to redirect flux to a bright output.

    The phase shifters are modeled as the input phases (one per input). The objective is
    to minimize the null-depth-like metric:
    
    Note: n_clad is calculated as n_clad = n_core - delta_n

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
    N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution :
        Same meaning as in :func:`simulate`. n_eff is computed automatically from n_core and n_clad.
    bright_output_idx : int, default=0
        Output index to maximize (the "Bright" output).
    Din, Dout : float, optional
        Input/output port spacing [m]. See :func:`simulate`.
    Sin, Sout : float, optional
        Input/output waveguide widths [m]. See :func:`simulate`.
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
    # Calculate n_clad from delta_n
    n_clad = n_core - delta_n
    
    if not (0 <= bright_output_idx < M):
        raise ValueError(f"bright_output_idx must be in [0, {M-1}], got {bright_output_idx}.")

    # Compute n_eff automatically from n_core and n_clad
    # Simple averaging for fundamental mode (weighted by core fraction)
    n_eff = 0.7 * n_core + 0.3 * n_clad  # Typical weighting for fundamental mode
    
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
            n_core=n_core,
            delta_n=delta_n,
            wavelength=wavelength,
            input_amplitudes=amps,
            num_modes=num_modes,
            num_z_steps=num_z_steps,
            z_resolution=z_resolution,
            output_file=None,
            verbose=False,
            Din=Din,
            Dout=Dout,
            Sin=Sin,
            Sout=Sout,
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


def calibrate_n_core_and_phases(
    N=4,
    M=4,
    L=None,
    W=10.0e-6,
    n_core_min=None,
    n_core_max=None,
    n_core_initial=2.0458,
    delta_n=0.0958,
    wavelength=1.55e-6,
    input_amplitudes=None,
    bright_output_idx=0,
    num_modes=50,
    num_z_steps=None,
    z_resolution=None,
    Din=None,
    Dout=None,
    Sin=None,
    Sout=None,
    n_core_steps_coarse=20,
    gradient_convergence_threshold=1e-3,
    gradient_initial_step=0.01,
    beta=0.8,
    initial_step=np.pi / 2,
    epsilon=1e-4,
    verbose=False,
    progress_callback_coarse=None,
    progress_callback_gradient=None,
):
    """Calibrate n_core and input phases to optimize null depth using coarse scan + gradient descent.
    
    This function performs a hierarchical optimization strategy:
    
    **Stage 1: Coarse Scan**
        - Explores wide range from 1.0 to 2×n_core_initial
        - Identifies promising starting point with deep nulls
    
    **Stage 2: Gradient Descent**
        - Starting from best coarse point, descends gradient
        - Adaptive step size with convergence when |Δn_core| < threshold
        - Automatically stops when no further improvement possible
    
    This approach efficiently finds optimal n_core even when it lies far from
    the initial guess, then refines to high precision via gradient descent.
    
    Parameters
    ----------
    N, M, L, W, delta_n, wavelength, input_amplitudes, bright_output_idx, num_modes, num_z_steps, z_resolution :
        Same meaning as in :func:`calibrate_input_phases_genetic`.
    n_core_min : float, optional
        Minimum n_core for coarse scan. Defaults to 1.0.
    n_core_max : float, optional
        Maximum n_core for coarse scan. Defaults to 2 × n_core_initial.
    n_core_initial : float, default=2.0458
        Initial/center value for n_core (used to set default range).
    n_core_steps_coarse : int, default=20
        Number of n_core values in coarse scan.
    gradient_convergence_threshold : float, default=1e-3
        Stop gradient descent when |Δn_core| < this value.
    gradient_initial_step : float, default=0.01
        Initial step size for gradient descent.
    Din, Dout, Sin, Sout, beta, initial_step, epsilon :
        Same as in :func:`calibrate_input_phases_genetic`.
    verbose : bool
        Print progress.
    progress_callback_coarse : callable, optional
        Function called after each coarse scan evaluation: callback(current_step, total_steps).
    progress_callback_gradient : callable, optional
        Function called after each gradient step: callback(iteration, delta_n_core).
    
    Returns
    -------
    dict
        Dictionary with:
        - ``n_core_values_coarse``: array of coarse scan n_core values
        - ``metrics_coarse``: corresponding metrics for coarse scan
        - ``n_core_values_gradient``: list of n_core values visited during gradient descent
        - ``metrics_gradient``: corresponding metrics for gradient descent
        - ``best_n_core``: optimal n_core value (from gradient descent)
        - ``best_metric``: best null depth metric achieved
        - ``best_phases``: optimal phases [rad] for the best n_core
        - ``bright_output_idx``: bright output index
    """
    # Set default search range for coarse scan
    if n_core_min is None:
        n_core_min = 1.0
    if n_core_max is None:
        n_core_max = 2.0 * n_core_initial
    
    if n_core_min >= n_core_max:
        raise ValueError(f"n_core_min ({n_core_min}) must be < n_core_max ({n_core_max})")
    
    if n_core_steps_coarse < 2:
        raise ValueError(f"n_core_steps_coarse must be >= 2, got {n_core_steps_coarse}")
    
    # ========================================================================
    # STAGE 1: COARSE SCAN
    # ========================================================================
    if verbose:
        print("="*70)
        print("STAGE 1: COARSE N_CORE SCAN")
        print("="*70)
        print(f"n_core range: [{n_core_min:.4f}, {n_core_max:.4f}]")
        print(f"n_core steps: {n_core_steps_coarse}")
        print(f"Bright output: {bright_output_idx}")
        print("="*70)
    
    n_core_values_coarse = np.linspace(n_core_min, n_core_max, n_core_steps_coarse)
    metrics_coarse = []
    all_phases_coarse = []
    
    for i, n_core_test in enumerate(n_core_values_coarse):
        if verbose:
            print(f"\n[Coarse {i+1}/{n_core_steps_coarse}] Testing n_core = {n_core_test:.4f}")
        
        # Calibrate phases for this n_core
        result = calibrate_input_phases_genetic(
            N=N,
            M=M,
            L=L,
            W=W,
            n_core=n_core_test,
            delta_n=delta_n,
            wavelength=wavelength,
            input_amplitudes=input_amplitudes,
            bright_output_idx=bright_output_idx,
            num_modes=num_modes,
            num_z_steps=num_z_steps,
            z_resolution=z_resolution,
            Din=Din,
            Dout=Dout,
            Sin=Sin,
            Sout=Sout,
            beta=beta,
            initial_step=initial_step,
            epsilon=epsilon,
            verbose=False,
        )
        
        metric = result['best_metric']
        phases = result['best_phases']
        
        metrics_coarse.append(metric)
        all_phases_coarse.append(phases)
        
        if progress_callback_coarse is not None:
            progress_callback_coarse(i + 1, n_core_steps_coarse)
        
        if verbose:
            print(f"   Metric: {metric:.3e}")
    
    metrics_coarse = np.array(metrics_coarse)
    best_coarse_idx = np.argmin(metrics_coarse)
    best_coarse_n_core = n_core_values_coarse[best_coarse_idx]
    best_coarse_metric = metrics_coarse[best_coarse_idx]
    
    if verbose:
        print("\n" + "="*70)
        print("COARSE SCAN COMPLETE")
        print("="*70)
        print(f"Best coarse n_core: {best_coarse_n_core:.4f}")
        print(f"Best coarse metric: {best_coarse_metric:.3e}")
        print("="*70)
    
    # ========================================================================
    # STAGE 2: GRADIENT DESCENT
    # ========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STAGE 2: GRADIENT DESCENT")
        print("="*70)
        print(f"Starting from: n_core = {best_coarse_n_core:.4f}")
        print(f"Convergence threshold: |Δn_core| < {gradient_convergence_threshold}")
        print("="*70)
    
    # Helper function to evaluate metric at a given n_core
    def evaluate_n_core(n_core_val):
        result = calibrate_input_phases_genetic(
            N=N,
            M=M,
            L=L,
            W=W,
            n_core=n_core_val,
            delta_n=delta_n,
            wavelength=wavelength,
            input_amplitudes=input_amplitudes,
            bright_output_idx=bright_output_idx,
            num_modes=num_modes,
            num_z_steps=num_z_steps,
            z_resolution=z_resolution,
            Din=Din,
            Dout=Dout,
            Sin=Sin,
            Sout=Sout,
            beta=beta,
            initial_step=initial_step,
            epsilon=epsilon,
            verbose=False,
        )
        return result['best_metric'], result['best_phases']
    
    # Initialize gradient descent
    n_core_current = best_coarse_n_core
    metric_current = best_coarse_metric
    phases_current = all_phases_coarse[best_coarse_idx]
    step_size = gradient_initial_step
    
    n_core_values_gradient = [n_core_current]
    metrics_gradient = [metric_current]
    all_phases_gradient = [phases_current]
    
    iteration = 0
    max_iterations = 100  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        
        # Evaluate gradient by finite differences
        n_core_plus = n_core_current + step_size
        n_core_minus = n_core_current - step_size
        
        # Clip to valid range
        n_core_plus = np.clip(n_core_plus, n_core_min, n_core_max)
        n_core_minus = np.clip(n_core_minus, n_core_min, n_core_max)
        
        if verbose:
            print(f"\n[Gradient {iteration}] Evaluating gradient at n_core = {n_core_current:.4f}")
        
        metric_plus, phases_plus = evaluate_n_core(n_core_plus)
        metric_minus, phases_minus = evaluate_n_core(n_core_minus)
        
        if verbose:
            print(f"   n_core={n_core_minus:.4f} → metric={metric_minus:.3e}")
            print(f"   n_core={n_core_plus:.4f}  → metric={metric_plus:.3e}")
        
        # Determine best direction
        if metric_plus < metric_current and metric_plus < metric_minus:
            # Move in + direction
            n_core_new = n_core_plus
            metric_new = metric_plus
            phases_new = phases_plus
            direction = "+"
        elif metric_minus < metric_current and metric_minus < metric_plus:
            # Move in - direction
            n_core_new = n_core_minus
            metric_new = metric_minus
            phases_new = phases_minus
            direction = "-"
        else:
            # No improvement, reduce step size
            step_size *= 0.5
            if verbose:
                print(f"   No improvement. Reducing step size to {step_size:.4f}")
            
            # Check convergence
            if step_size < gradient_convergence_threshold:
                if verbose:
                    print(f"   Step size below threshold. Converged!")
                break
            continue
        
        delta_n_core = abs(n_core_new - n_core_current)
        
        if verbose:
            print(f"   → Moving {direction}: n_core = {n_core_new:.4f}, Δn = {delta_n_core:.4f}")
        
        # Update current position
        n_core_current = n_core_new
        metric_current = metric_new
        phases_current = phases_new
        
        n_core_values_gradient.append(n_core_current)
        metrics_gradient.append(metric_current)
        all_phases_gradient.append(phases_current)
        
        if progress_callback_gradient is not None:
            progress_callback_gradient(iteration, delta_n_core)
        
        # Check convergence
        if delta_n_core < gradient_convergence_threshold:
            if verbose:
                print(f"   Converged! |Δn_core| = {delta_n_core:.4f} < {gradient_convergence_threshold}")
            break
    
    if iteration >= max_iterations:
        if verbose:
            print(f"\n   Warning: Max iterations ({max_iterations}) reached without convergence")
    
    best_n_core = n_core_current
    best_metric = metric_current
    best_phases = phases_current
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Gradient iterations: {len(n_core_values_gradient) - 1}")
        print(f"Optimal n_core: {best_n_core:.4f}")
        print(f"Best metric (null/bright): {best_metric:.3e}")
        print(f"Best phases [rad]: {best_phases}")
        print("="*70)
    
    return {
        "n_core_values_coarse": n_core_values_coarse,
        "metrics_coarse": metrics_coarse,
        "n_core_values_gradient": np.array(n_core_values_gradient),
        "metrics_gradient": np.array(metrics_gradient),
        "best_n_core": best_n_core,
        "best_metric": best_metric,
        "best_phases": best_phases,
        "bright_output_idx": bright_output_idx,
    }


def _compute_mode_profile(x_grid, center, width):
    """Compute normalized Gaussian mode profile for a single-mode waveguide.
    
    This function models the fundamental mode of a single-mode step-index waveguide
    with a Gaussian approximation. The mode is assumed to be centered at position
    `center` with effective width (Field Mode Width) of `width`.
    
    Physical reasoning:
    - Single-mode waveguides confine light via total internal reflection
    - The fundamental LP₀₁ mode is approximately Gaussian in transverse profile
    - The width parameter represents the 1/e² intensity radius of this mode
    
    Parameters
    ----------
    x_grid : np.ndarray
        Spatial grid points along the transverse direction [m].
    center : float
        Center position of the mode along x [m].
    width : float
        Effective mode width (1/e² intensity radius) [m].
        For a step-index fiber, this relates to the V-parameter and mode confinement.
    
    Returns
    -------
    np.ndarray
        Normalized mode profile ψ(x) such that ∫|ψ(x)|² dx = 1.
        Units: [1/√m] (inverse square root of length, normalized).
        
    Notes
    -----
    The Gaussian approximation is standard in integrated photonics and fiber optics:
    - ψ(x) ∝ exp(-(x - center)² / (width/2)²)
    - Normalization ensures energy conservation
    """
    if width <= 0:
        raise ValueError(f"Mode width must be > 0, got {width}.")
    
    # Gaussian profile: exp(-(x - center)² / σ²)
    # where σ = width/2 (so that 1/e² radius = width)
    sigma = width / 2.0
    profile = np.exp(-((x_grid - center)**2) / (sigma**2))
    
    # Normalize: ∫|ψ|² dx = 1
    # Using trapezoidal rule integration
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    norm_factor = np.sqrt(np.sum(np.abs(profile)**2) * dx)
    
    if norm_factor <= 0:
        raise ValueError("Mode profile normalization failed (zero or negative norm).")
    
    return profile / norm_factor


def _compute_symmetric_port_positions(num_ports, W, spacing, name):
    """Compute symmetric port positions centered at x=0.

    Parameters
    ----------
    num_ports : int
        Number of ports.
    W : float
        MMI width [m]. The MMI core extends from [-W/2, W/2].
    spacing : float | None
        Port-to-port spacing [m]. If None, uses the historical default spacing W/num_ports.
    name : str
        Human-readable name used for error messages (e.g., "input", "output").

    Returns
    -------
    list[float]
        Port center positions along x in [m], symmetric about x=0 (centered).

    Raises
    ------
    ValueError
        If spacing is non-positive or causes ports to lie outside the MMI [-W/2, W/2].
    """
    if num_ports <= 0:
        raise ValueError(f"{name} ports must be a positive integer, got {num_ports}.")
    if W <= 0:
        raise ValueError(f"MMI width W must be positive, got {W}.")

    if spacing is None:
        spacing = W / num_ports
    if spacing <= 0:
        raise ValueError(f"{name} spacing must be > 0, got {spacing}.")

    # Center at x=0, distribute symmetrically
    center = 0.0
    offsets = (np.arange(num_ports, dtype=float) - 0.5 * (num_ports - 1)) * spacing
    positions = center + offsets

    # Numerical tolerance in meters (scaled with W).
    eps = max(1e-15, 1e-15 * abs(W))
    min_pos = float(np.min(positions))
    max_pos = float(np.max(positions))
    if (min_pos < -W/2 - eps) or (max_pos > W/2 + eps):
        raise ValueError(
            f"{name} spacing {spacing} m is too large for W={W} m: "
            f"{name} positions would span [{min_pos}, {max_pos}] m outside [-{W/2}, {W/2}] m."
        )

    # Clamp tiny numerical noise at boundaries.
    positions = np.clip(positions, -W/2, W/2)
    return positions.tolist()


def _solve_slab_modes_fd(x_grid, n_profile, k0, num_modes):
    """Solve 1D slab modes with finite differences (Dirichlet at boundaries).

    We shift the potential by the cladding term to keep eigenvalues well-scaled:
    solve ``-d²ψ/dx² + [(k0 n)^2 - (k0 n_clad)^2] ψ = (β² - (k0 n_clad)^2) ψ``.

    Returns modes sorted by β (high to low) and normalized (∫|ψ|² dx = 1).
    """
    dx = x_grid[1] - x_grid[0]
    n_pts = len(x_grid)

    main = np.full(n_pts, -2.0 / dx**2)
    off = np.full(n_pts - 1, 1.0 / dx**2)
    lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    n_clad = float(np.min(n_profile))
    potential = (k0 * n_profile) ** 2 - (k0 * n_clad) ** 2
    A = -lap + np.diag(potential)

    eigvals, eigvecs = np.linalg.eigh(A)
    betas = np.sqrt(np.clip(eigvals + (k0 * n_clad) ** 2, 0.0, None))

    # Sort by beta descending (guided first)
    idx = np.argsort(betas)[::-1]
    betas = betas[idx][:num_modes]
    modes = eigvecs[:, idx][:, :num_modes].T  # (num_modes, n_pts)

    # Normalize modes
    for m in range(len(modes)):
        norm = np.sqrt(np.trapz(np.abs(modes[m])**2, x_grid))
        if norm > 0:
            modes[m] /= norm

    return betas, modes


def _propagate_free_space(input_field, x_grid, z_grid, k0, n_medium):
    """Propagate a field in a uniform medium using the angular spectrum method.

    This fallback is used when the index contrast collapses (Δn → 0), so the
    waveguide no longer supports guided modes and the field should simply
    diffract in free space.
    """
    dx = x_grid[1] - x_grid[0]
    kx = 2 * np.pi * np.fft.fftfreq(len(x_grid), d=dx)
    k_cutoff = k0 * n_medium

    # Split propagating vs evanescent components for numerical stability
    kx_abs = np.abs(kx)
    kz = np.zeros_like(kx, dtype=complex)
    propagating = kx_abs <= k_cutoff
    kz[propagating] = np.sqrt(np.maximum(k_cutoff**2 - kx[propagating]**2, 0.0))
    kz[~propagating] = 1j * np.sqrt(kx[~propagating]**2 - k_cutoff**2)

    spectrum0 = np.fft.fft(input_field)
    field_evolution = np.zeros((len(z_grid), len(x_grid)), dtype=complex)

    for iz, z in enumerate(z_grid):
        phase = np.exp(1j * kz * z)
        field_evolution[iz, :] = np.fft.ifft(spectrum0 * phase)

    return field_evolution

def _compute_mmi_field(N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose=False, Din=None, Dout=None, Sin=None, Sout=None):
    """
    Core field calculation (Internal helper).
    
    Parameters
    ----------
    N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din, Dout : 
        Standard MMI simulation parameters. n_eff is computed automatically from n_core and delta_n for each mode.
        n_clad = n_core - delta_n
    Sin : float, optional
        Width of input waveguide single-mode field (Field Mode Width, FMW) [m].
        If None, uses historical default: (W / N) / 4.
        This parameter defines how the input field couples from the monomode input waveguide
        into the multi-mode interference region.
    Sout : float, optional
        Width of output waveguide single-mode field (FMW) [m].
        If None, uses historical default: (W / N) / 4.
        This parameter defines the effective coupling aperture at each output.
    
    Returns
    -------
    tuple
        (z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx)
        where beam_waist is the effective mode width used for overlaps.
    """
    # Calculate n_clad from delta_n
    n_clad = n_core - delta_n
    
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
    
    # 2. Define Waveguide Modes
    # Use sine modes basis with physical confinement via Δn
    # Simulation window extends from -W to W (total width = 2W) to capture evanescent decay
    # MMI region itself is centered at x=0: [-W/2, W/2]
    x_grid = np.linspace(-W, W, 500)
    dx = x_grid[1] - x_grid[0]
    
    betas = []
    n_eff_modes = []
    modes = []

    for m in range(num_modes):
        # Mode number m (starting from 1 for m=0)
        mode_num = m + 1
        
        # Wave vector in transverse direction (sine profile in core)
        kx_m = mode_num * np.pi / W
        
        # Propagation constant: β² = (k₀ n_core)² - kx²
        sq_term = (k0 * n_core) ** 2 - kx_m ** 2
        
        if sq_term < 0:
            # Mode is beyond cutoff
            betas.append(0.0)
            n_eff_modes.append(n_clad)
            modes.append(np.zeros_like(x_grid, dtype=float))
            continue
        
        beta_m = np.sqrt(sq_term)
        betas.append(beta_m)
        
        # Effective refractive index for this mode
        n_eff_m = beta_m / k0
        n_eff_modes.append(n_eff_m)
        
        # Construct mode profile
        phi_m = np.zeros_like(x_grid, dtype=float)
        
        # Inside core [-W/2, W/2]: sine profile (normalized)
        # Shift coordinate system: x_core = x + W/2 to map [-W/2, W/2] to [0, W] for sine
        mask_inside = (x_grid >= -W/2) & (x_grid <= W/2)
        x_shifted = x_grid[mask_inside] + W/2
        phi_m[mask_inside] = np.sqrt(2 / W) * np.sin(kx_m * x_shifted)
        
        # Outside core: evanescent decay with proper decay constant
        # Decay constant κ = sqrt(kx² - (k₀ n_clad)²)
        kx_clad_sq = (k0 * n_clad) ** 2
        sq_decay = kx_m ** 2 - kx_clad_sq
        
        if sq_decay > 0:
            # Evanescent region
            kappa_m = np.sqrt(sq_decay)
            
            # Left side (x < -W/2): decay as exp(κ(x + W/2))
            mask_left = x_grid < -W/2
            phi_m[mask_left] = np.sqrt(2 / W) * np.exp(kappa_m * (x_grid[mask_left] + W/2))
            
            # Right side (x > W/2): decay as exp(-κ(x - W/2))
            mask_right = x_grid > W/2
            phi_m[mask_right] = np.sqrt(2 / W) * np.exp(-kappa_m * (x_grid[mask_right] - W/2))
        else:
            # Oscillating region (if core index < clad index, which shouldn't happen)
            # For safety, just use a smooth cutoff
            mask_left = x_grid < -W/2
            phi_m[mask_left] = np.sqrt(2 / W) * np.cos(kx_m * (x_grid[mask_left] + W/2)) * np.exp(-np.abs(x_grid[mask_left] + W/2) / (W / 10))
            
            mask_right = x_grid > W/2
            phi_m[mask_right] = np.sqrt(2 / W) * np.cos(kx_m * (x_grid[mask_right] - W/2)) * np.exp(-(x_grid[mask_right] - W/2) / (W / 10))
        
        modes.append(phi_m)
    
    betas = np.array(betas)
    n_eff_modes = np.array(n_eff_modes)
    modes = np.array(modes)  # Shape: (num_modes, num_x_points)

    if verbose:
        print(f"\nMode-Dependent Effective Indices:")
        print(f"{'Mode':>6s} {'n_eff':>10s} {'Type':>12s}")
        print("-" * 50)
        for m in range(min(8, num_modes)):
            mode_type = "guided" if betas[m] > 0 else "cutoff"
            print(f"LP₁{m+1:>2d} {n_eff_modes[m]:>10.4f} {mode_type:>12s}")
        if num_modes > 8:
            print(f"... ({num_modes - 8} more modes)")
    
    # 3. Construct Input Field (Gaussian beams from single-mode input waveguides)
    input_field = np.zeros_like(x_grid, dtype=complex)
    
    # Determine input mode width (Si = "S input")
    # If not provided, use historical default
    if Sin is None:
        Sin = (W / N) / 4
    
    if verbose:
        print(f"\nInput mode width (Sin) = {Sin*1e6:.3f} um")
        print(f"Input port positions [m] (centered at x=0): {input_positions}")
        print(f"Input port positions [um] (centered at x=0): {[p*1e6 for p in input_positions]}")
    
    if verbose:
        print(f"Injecting input vector: {input_amplitudes}")
    
    for idx, amp in enumerate(input_amplitudes):
        if amp == 0:
            continue
        center_x = input_positions[idx]
        # Use the new _compute_mode_profile function for input coupling
        gauss = _compute_mode_profile(x_grid, center_x, Sin)
        input_field += amp * gauss
        if verbose and idx < 2:  # Print first 2 for debugging
            print(f"  Input {idx}: center={center_x*1e6:.3f} um, amplitude={amp}")
    
    # Store Sin as beam_waist for later overlap calculations
    beam_waist = Sin

    # Prepare z grid for propagation
    z_grid = np.linspace(0, L, num_z_steps)

    # 4. Mode Decomposition
    coeffs = np.array([
        np.trapz(input_field * np.conj(modes[m]), x_grid) for m in range(num_modes)
    ])
    
    # 5. Propagation with mode-dependent n_eff
    field_evolution = np.zeros((num_z_steps, len(x_grid)), dtype=complex)
    
    iterator = enumerate(z_grid)
    if verbose:
        # Use tqdm for the loop
        iterator = enumerate(tqdm(z_grid, desc="Simulating Propagation", unit="step"))
        
    for iz, z in iterator:
        # Use mode-dependent effective indices for phase evolution
        phase_term = np.exp(-1j * betas * z)  # betas computed from n_core
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
    
    # 1. Static Plot (CENTERED COORDINATES)
    extent = [0, L*1e6, -W*1e6, W*1e6]  # z: [0, L], x: [-W, W]
    ax_static.set_title(f"MMI Propagation Field Intensity (L={L*1e6:.1f}um, W={W*1e6:.1f}um)")
    ax_static.imshow(intensity_evolution.T, origin='lower', aspect='auto', 
                          extent=extent, cmap='inferno')
    ax_static.set_xlabel("z [um]")
    ax_static.set_ylabel("x [um]")
    
    # MMI core boundaries [-W/2, W/2]
    ax_static.axhline(y=-W/2*1e6, color='white', linestyle=':', linewidth=1, alpha=0.5)
    ax_static.axhline(y=W/2*1e6, color='white', linestyle=':', linewidth=1, alpha=0.5)
    ax_static.axhline(y=0, color='cyan', linestyle='-', linewidth=1.5, alpha=0.7, label='x=0')
    
    # Inputs/Outputs markers
    for y_pos in input_positions:
        ax_static.text(0, y_pos*1e6, 'In', color='white', ha='right', va='center', fontsize=8)
    for y_pos in output_positions:
        ax_static.text(L*1e6, y_pos*1e6, 'Out', color='white', ha='left', va='center', fontsize=8)

    # Moving vertical line
    ax_static.plot([z_val*1e6, z_val*1e6], [-W*1e6, W*1e6], 'w--', lw=1.5)
    
    # 2. Dynamic Plot
    ax_anim.set_title(f"Cross-section at z = {z_val*1e6:.1f} um")
    ax_anim.set_xlim(-W*1e6, W*1e6)
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
        
    # -- Static Plot (CENTERED COORDINATES) --
    extent = [0, L*1e6, -W*1e6, W*1e6]  # z: [0, L], x: [-W, W]
    ax_static.set_title(f"Field Intensity (L={L*1e6:.1f}um)")
    ax_static.imshow(intensity_total_evol.T, origin='lower', aspect='auto', extent=extent, cmap='inferno')
    ax_static.set_xlabel("z [um]")
    ax_static.set_ylabel("x [um]")
    ax_static.axhline(y=0, color='cyan', linestyle='-', linewidth=1.5, alpha=0.7)  # Center line
    ax_static.plot([z_val*1e6, z_val*1e6], [-W*1e6, W*1e6], 'w--', lw=1.5)
    
    # Markers
    ax_static.scatter([0]*N, [p*1e6 for p in input_positions], color='white', marker='o', s=20, zorder=10)
    ax_static.scatter([L*1e6]*M, [p*1e6 for p in output_positions], color='white', marker='o', s=20, zorder=10)

    # -- Profile Plot --
    ax_anim.set_title(f"Cross-section at z={z_val*1e6:.1f} um")
    ax_anim.set_xlim(-W*1e6, W*1e6)
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

def _compute_input_intensity_normalization(field_z0, x_grid, W, dx):
    """
    Compute input intensity normalization factor from MMI CORE region only.
    
    Calculates intensity integrated over the MMI core region [0.5*W, 1.5*W] at z=0.
    This normalization makes the core power equal to 1.0 at the input plane.
    
    **Physical interpretation after normalization:**
    
    - At z=0: Core power = 1.0 (by construction)
    - During propagation:
      - core_power > 1.0: Energy from outside core couples in (evanescent → core)
      - core_power < 1.0: Energy leaks from core to evanescent regions
      - core_power = 1.0: Energy remains confined in core
    
    **Important note:** At z=0, if the input modes are correctly positioned at the
    entrance of the MMI core, there should be very little power outside the core region
    (<10%). The evanescent tails develop during propagation inside the MMI, not at z=0.
    
    If significant power (>10%) is outside the core at z=0, this indicates:
    - Input mode field diameter (MFD) too large
    - Input waveguide positions incorrect
    - Input modes already include propagation effects
    
    This function will issue a warning if more than 10% of power is outside core at z=0.
    
    Parameters
    ----------
    field_z0 : array
        Complex electric field at z=0 (first z-plane).
    x_grid : array
        x spatial grid points [m].
    W : float
        Width of the MMI region [m]. Core is [0.5*W, 1.5*W].
    dx : float
        Spatial grid spacing [m].
    
    Returns
    -------
    float
        Input power integrated over MMI core region. If zero or near-zero,
        returns 1.0 to avoid division issues.
    """
    intensity_z0 = np.abs(field_z0)**2
    
    # Integrate over MMI CORE region only [-W/2, W/2] (centered at x=0)
    mask = (x_grid >= -W/2) & (x_grid <= W/2)
    input_power_core = np.sum(intensity_z0[mask]) * dx
    
    # For diagnostic: compute total power and fraction outside core
    input_power_total = np.sum(intensity_z0) * dx
    power_outside_core = input_power_total - input_power_core
    fraction_outside = power_outside_core / input_power_total if input_power_total > 1e-12 else 0.0
    
    # Warn if significant power is outside core at z=0 (should not happen with proper input modes)
    if fraction_outside > 0.10:  # More than 10% outside
        import warnings
        warnings.warn(
            f"⚠️ Significant power outside MMI core at z=0: {fraction_outside*100:.1f}%\n"
            f"   Power in core: {input_power_core:.3e}, Power total: {input_power_total:.3e}\n"
            f"   This suggests input modes are too wide or incorrectly positioned.\n"
            f"   Expected: <10% outside core at z=0 (evanescent tails develop during propagation)",
            UserWarning
        )
    
    # Avoid division by zero
    if input_power_core < 1e-15:
        return 1.0
    
    return input_power_core


def _compute_single_field_wrapper(i, N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, Din, Dout, Sin, Sout):
    """Wrapper to compute field for a single input (parallel helper)."""
    single_input = np.zeros(N, dtype=complex)
    single_input[i] = input_amplitudes[i]
    
    # We only need the field_evolution (3rd return, index 2)
    ret = _compute_mmi_field(
        N, M, L, W, n_core, delta_n, wavelength, single_input, num_modes, num_z_steps, z_resolution, verbose=False, Din=Din, Dout=Dout, Sin=Sin, Sout=Sout
    )
    return ret[2]

def simulate(N=2, M=2, L=None, W=10.0e-6, n_core=2.0458, delta_n=0.0958, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=None, z_resolution=None, output_file=None, verbose=False, Din=None, Dout=None, Sin=None, Sout=None):

    # Calculate n_clad from delta_n
    n_clad = n_core - delta_n

    """
    Simulates light propagation in an NxM MMI (Multi-Mode Interferometer) using Eigenmode Expansion with
    finite-difference step-index modes.
    
    This function models the propagation of light through a multimode waveguide section, calculating
    the output amplitudes at specified output ports. Modes are obtained from a finite-difference eigen
    solver on a step-index profile, capturing evanescent tails without the hard-wall approximation.

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
    n_core : float, default=2.0458
        Refractive index of the MMI core.
    n_clad : float, default=1.95
        Refractive index of the cladding (surrounding material).
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
    Sin : float, optional
        **Core diameter (d_core) of the input single-mode waveguides** [m].
        
        This is the PHYSICAL core diameter of the waveguide, not the Mode Field Width (MFD).
        The Mode Field Width is calculated internally using the Marcuse formula.
        
        If None, defaults to (W / N) / 4.
        
        **Important Distinctions:**
        - **d_core** (Sin parameter): The actual core geometry of the waveguide
        - **MFD**: Where the LP₀₁ mode concentrates (MFD ≈ d_core × Marcuse_factor)
        - **V-number**: V = (π·d_core/λ)·NA determines the number of modes
        
        **Example** (Silicon photonics @ 1.55 µm):
        - d_core = 0.5 µm → V ≈ 0.34 → Single-mode ✓
        - d_core = 2.0 µm → V ≈ 1.35 → Single-mode ✓
        - d_core = 3.0 µm → V ≈ 2.03 → Single-mode (barely)
        - d_core = 4.0 µm → V ≈ 2.71 → Weakly multimode ⚠️
        
    Sout : float, optional
        **Core diameter (d_core) of the output single-mode waveguides** [m].
        
        Same physical meaning as Sin, but for the output ports. If None, defaults to Sin
        (or (W / N) / 4 if Sin is also None).
        
        **For optimal nulling interferometry:**
        Keep Sout in single-mode regime (V < 2.405) to avoid modal noise.
        For typical Δn ≈ 0.1 and λ = 1.55 µm:
        - Sout < ~2.7 µm → Single-mode ✓
        - Sout > ~3.8 µm → Strongly multimode ❌
    output_file : str, optional
        Path to save an animation of the propagation (e.g., 'mmi_prop.mp4'). 
        If None, no animation is generated.
    verbose : bool, default=False
        If True, prints detailed simulation progress and parameters.

    Returns
    -------
    np.ndarray
        Complex amplitudes at the M output ports.
    
    Notes
    -----
    The simulation domain extends from -W/2 to 3W/2 (total width = 2W) to allow evanescent
    field loss at the boundaries with no reflections. The MMI region itself spans [0, W].
    """
    
    # Defaults logic        
    if L is None:
        # Compute n_eff automatically from n_core and n_clad
        n_eff = 0.7 * n_core + 0.3 * n_clad
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

    # Default output width to input width if not specified
    if Sout is None:
        Sout = Sin  # Will inherit Sin's default if Sin is also None
    
    # Run internal simulation
    z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx = _compute_mmi_field(
        N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din=Din, Dout=Dout, Sin=Sin, Sout=Sout
    )
    
    # Update num_z_steps to match the actual grid size used
    num_z_steps = len(z_grid)

    intensity_evolution = np.abs(field_evolution)**2
    
    # Normalize by input intensity (integrated over core region at z=0)
    input_power = _compute_input_intensity_normalization(field_evolution[0, :], x_grid, W, dx)
    if input_power > 0:
        intensity_evolution = intensity_evolution / input_power

    # --- Calculation of Output Vector (Multi-Mode Waveguide Coupling) ---
    # The output amplitudes are calculated by overlapping the MMI field at the output
    # plane (z=L) with ALL guided modes of each output waveguide, not just LP₀₁.
    #
    # Physical principle (rigorous treatment):
    # - Each output port is connected to a waveguide of core diameter Sout
    # - The V-number determines how many modes are guided: V = (π·a/λ)·NA
    # - For V < 2.405: Single-mode (only LP₀₁ couples)
    # - For V > 2.405: Multi-mode (LP₀₁, LP₁₁, LP₂₁, ... all couple)
    # - The total coupled power is: P_total = Σ_modes |∫ E(x,L) · ψ_mode(x) dx|²
    #
    # Key insight:
    # When Sout is LARGE, the waveguide supports multiple modes. The MMI field
    # couples to ALL of them, distributing energy across LP₀₁, LP₁₁, etc.
    # This REDUCES the coupling to LP₀₁ compared to a single-mode waveguide!
    #
    # This is contrary to the naive "larger Sout = more overlap" intuition.
    # It's why fiber splicing requires precise diameter matching.
    #
    # References:
    # - Marcuse, D. (1977). "Loss analysis of single-mode fiber splices."
    # - Snyder & Love (2012). "Optical Waveguide Theory", Chapter 13.
    
    output_amplitudes = []
    final_field = field_evolution[-1, :]
    
    # Compute n_eff for output waveguide analysis  
    n_eff = 0.7 * n_core + 0.3 * n_clad
    
    # Determine output mode width
    Sout_use = Sout if Sout is not None else (Sin if Sin is not None else (W / N) / 4)
    
    # Assume typical photonic waveguide indices
    n_core_out = n_core  # Use MMI core index for output waveguide
    n_cladding_out = n_clad  # Use MMI cladding index for output waveguide
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"OUTPUT WAVEGUIDE COUPLING ANALYSIS")
        print(f"{'='*60}")
        print(f"Output core diameter (Sout = d_core) = {Sout_use*1e6:.3f} µm")
        print(f"(NOTE: Sout is the PHYSICAL core diameter, not the Mode Field Width)")
        print(f"       Mode Field Width is calculated internally using Marcuse formula")
    
    # Check V-number and warn if multimode
    if _HAS_LP_MODES:
        V = compute_v_number(Sout_use, wavelength, n_core_out, n_cladding_out)
        
        if verbose:
            print(f"V-number = {V:.3f}")
            
            if V < 2.405:
                print("✓ SINGLE-MODE regime (V < 2.405)")
                print("  → Only LP₀₁ couples → optimal for nulling")
            elif V < 3.832:
                print("⚠️ WEAKLY MULTIMODE regime (2.405 < V < 3.832)")
                print("  → LP₀₁ + LP₁₁ modes propagate")
                print("  → Coupling splits between modes")
                print(f"  → Consider reducing Sout to < {2.405*wavelength/(np.pi*np.sqrt(2*0.1))*1e6:.2f} µm")
            else:
                print("❌ STRONGLY MULTIMODE regime (V > 3.832)")
                print("  → Multiple modes (LP₀₁, LP₁₁, LP₂₁, ...) propagate")
                print("  → SEVERE coupling degradation to fundamental mode")
                print(f"  → RECOMMENDED: Reduce Sout to < {2.405*wavelength/(np.pi*np.sqrt(2*0.1))*1e6:.2f} µm")
            print()
    
    # Compute coupling for each output
    for j in range(M):
        center_x_out = output_positions[j]
        
        if _HAS_LP_MODES and V > 2.405:
            # RIGOROUS: Compute multimode coupling
            coupling_data = compute_multimode_coupling(
                final_field,
                x_grid,
                center_x_out,
                Sout_use,
                wavelength,
                n_core_out,
                n_cladding_out,
                max_modes=5,
            )
            
            # The total coupled amplitude is the coherent sum of all mode couplings
            # For simplicity, we use sqrt(total_coupling) as the amplitude
            # (This is an approximation; rigorous treatment requires phase tracking)
            total_coupling_power = coupling_data['total_coupling']
            output_amp = np.sqrt(total_coupling_power)
            
            # Preserve phase from LP₀₁ dominant mode
            lp01_coupling = coupling_data['modes'][0]['coupling'] if coupling_data['modes'] else 0
            if lp01_coupling > 1e-10:
                # Compute LP01 overlap to get phase
                psi_lp01 = _compute_mode_profile(x_grid, center_x_out, Sout_use)
                overlap_lp01 = np.sum(final_field * np.conj(psi_lp01)) * dx
                phase = np.angle(overlap_lp01)
                output_amp = output_amp * np.exp(1j * phase)
            
            if verbose and j == 0:
                # Print detailed mode breakdown for first output
                print(f"Output #{j+1} - Multimode Coupling Breakdown:")
                print(f"  Total coupling efficiency: {total_coupling_power:.4f}")
                for mode_info in coupling_data['modes']:
                    fraction = mode_info['coupling'] / total_coupling_power if total_coupling_power > 1e-10 else 0
                    print(f"    {mode_info['label']}: {mode_info['coupling']:.4f} ({fraction*100:.1f}%)")
                print()
                
        else:
            # SINGLE-MODE or fallback: Use Gaussian approximation
            psi_out = _compute_mode_profile(x_grid, center_x_out, Sout_use)
            overlap = np.sum(final_field * np.conj(psi_out)) * dx
            output_amp = overlap
        
        output_amplitudes.append(output_amp)
        
    output_amplitudes = np.array(output_amplitudes)

    if verbose:
        print(f"Output amplitudes: {output_amplitudes}")
        print(f"Output intensities: {np.abs(output_amplitudes)**2}")
        print(f"{'='*60}\n")

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

def compute_contributions(N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose=False, Din=None, Dout=None, Sin=None, Sout=None):
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
    n_core : float
        Refractive index of the MMI core.
    delta_n : float
        Index contrast (n_core - n_clad). n_clad = n_core - delta_n.
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
    Sin : float, optional
        **Core diameter (d_core) of the input single-mode waveguides** [m].
        See :func:`simulate` for detailed description of physical meaning.
    Sout : float, optional
        **Core diameter (d_core) of the output single-mode waveguides** [m].
        See :func:`simulate` for detailed description of physical meaning.

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
    # Calculate n_clad from delta_n
    n_clad = n_core - delta_n
    
    if L is None:
        n_eff = 0.7 * n_core + 0.3 * n_clad
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2
        if verbose:
            print(f"Auto-calculated L = {L*1e6:.2f} um")
    if input_amplitudes is None:
        val = 1.0 / np.sqrt(N)
        input_amplitudes = [val] * N
    if len(input_amplitudes) != N:
        raise ValueError(f"Length mismatch")

    # Default output width to input width if not specified
    if Sout is None:
        Sout = Sin

    # 1. Individual Simulations
    contributions_fields = [] # List of N arrays (num_z, num_x)
    
    # Common geometry data from first run
    z_grid, x_grid, field_total_evol, output_positions, input_positions, beam_waist, dx = _compute_mmi_field(
        N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din=Din, Dout=Dout, Sin=Sin, Sout=Sout
    )
    
    # Update num_z_steps to match (important for animation frames)
    num_z_steps = len(z_grid)
    
    # Store total intensity for main plot
    intensity_total_evol = np.abs(field_total_evol)**2
    
    # Normalize by input intensity (integrated over core region at z=0)
    input_power = _compute_input_intensity_normalization(field_total_evol[0, :], x_grid, W, dx)
    if input_power > 0:
        intensity_total_evol = intensity_total_evol / input_power

    # Now compute individual contributions
    # For each input i, simulate with only input_amplitudes[i] active
    if verbose:
        print("Computing separate field contributions (Parallel)...")
    
    contributions_fields = Parallel(n_jobs=-1)(
        delayed(_compute_single_field_wrapper)(
            i, N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, Din, Dout, Sin, Sout
        ) for i in range(N)
    )
        
    # Pre-compute Overlaps (Phasors) for all z steps using output mode width Sout
    phasors = np.zeros((num_z_steps, M, N), dtype=complex)
    
    # Determine effective output mode width
    Sout_use = Sout if Sout is not None else (Sin if Sin is not None else (W / N) / 4)
    
    # Pre-compute normalized output modes
    output_modes = [] # shape (M, num_x)
    for j in range(M):
        center_x_out = output_positions[j]
        psi = _compute_mode_profile(x_grid, center_x_out, Sout_use)
        output_modes.append(psi)
        
    for iz in range(num_z_steps):
        for j in range(M): # Output J
            psi_out = output_modes[j]
            for i in range(N): # Input I
                E_i_z = contributions_fields[i][iz, :]
                # Overlap integral
                coupling = np.sum(E_i_z * np.conj(psi_out)) * dx
                phasors[iz, j, i] = coupling

    # Compute final Sin and Sout values (after defaults)
    Sin_final = Sin if Sin is not None else (W / N) / 4
    Sout_final = Sout_use

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
        "num_z_steps": num_z_steps,
        "Sin": Sin_final,
        "Sout": Sout_final
    }


def plot_mmi_interactive(
    N, M, L, W, n_core, delta_n, wavelength, input_amplitudes,
    num_modes, num_z_steps, z_resolution, Din, Dout, Sin, Sout,
    verbose=False
):
    """Generate interactive MMI visualization plot.
    
    Builds and returns a matplotlib figure showing the complete MMI simulation
    with intensity map, cross-sections, phasor contributions, and z-profiles.
    
    Parameters
    ----------
    N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes : 
        Same as :func:`compute_contributions`.
    num_z_steps, z_resolution, Din, Dout, Sin, Sout, verbose :
        Same as :func:`compute_contributions`.
    
    Returns
    -------
    matplotlib.figure.Figure
        The complete MMI visualization figure.
    """
    # Compute data
    data = compute_contributions(
        N=N, M=M, L=L, W=W, n_core=n_core, delta_n=delta_n,
        wavelength=wavelength, input_amplitudes=input_amplitudes,
        num_modes=num_modes, num_z_steps=num_z_steps, z_resolution=z_resolution,
        Din=Din, Dout=Dout, Sin=Sin, Sout=Sout, verbose=verbose
    )
    
    z_grid = data['z_grid']
    x_grid = data['x_grid']
    intensity_map = data['intensity_total_evol']
    phasors = data['phasors']
    input_pos = data['input_positions']
    output_pos = data['output_positions']
    L_sim = data['L']
    W = data['W']
    Sin_computed = data['Sin']
    Sout_computed = data['Sout']

    # Build figure
    fig = plt.figure(figsize=(12, 24))
    gs = fig.add_gridspec(6, M, height_ratios=[1.5, 1, 1, 1.5, 1.5, 1])

    # 1. Intensity Map (Top) - CENTERED COORDINATES
    ax_map = fig.add_subplot(gs[0, :])
    # Show full 2W simulation window (x_grid goes from -W to W, centered at x=0)
    x_min = x_grid[0]  # -W
    x_max = x_grid[-1]  # W
    extent = [0, L_sim*1e6, x_min*1e6, x_max*1e6]  # z: [0, L], x: [-W, W]
    
    im = ax_map.imshow(intensity_map.T, origin='lower', aspect='auto', extent=extent, cmap='inferno')
    ax_map.set_xlabel('z [um]')
    ax_map.set_ylabel('x [um] (centered at 0)')
    
    # Add lines to mark MMI core boundaries
    # MMI core: x ∈ [-W/2, W/2] (centered at x=0)
    ax_map.axhline(y=-W/2*1e6, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='MMI Core Boundary')
    ax_map.axhline(y=W/2*1e6, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Title with mode widths and wavelength
    sin_str = f"{Sin*1e6:.2f}" if Sin else "auto"
    sout_str = f"{Sout*1e6:.2f}" if Sout else "auto"
    n_clad_calc = n_core - delta_n
    n_eff_calc = 0.7 * n_core + 0.3 * n_clad_calc
    ax_map.set_title(f'Intensity Map - Centered Coords (λ={wavelength*1e6:.2f} µm, n_core={n_core:.4f}, Δn={delta_n:.4f}, n_eff={n_eff_calc:.4f})\nSin={sin_str} µm, Sout={sout_str} µm | White lines: MMI core [-W/2, W/2], Cyan: x=0', fontsize=9)
    
    # Markers - already in centered coordinates
    ax_map.scatter([0]*N, [p*1e6 for p in input_pos], color='cyan', s=10, marker='o', label='Inputs')
    ax_map.scatter([L_sim*1e6]*M, [p*1e6 for p in output_pos], color='lime', s=10, marker='s', label='Outputs')
    ax_map.legend(loc='upper right', fontsize=8)

    # 2. Input Profile (Row 2) - CENTERED COORDINATES
    ax_prof_in = fig.add_subplot(gs[1, :])
    # x_grid is already in centered coordinates [-W, W]
    x_display = x_grid * 1e6
    ax_prof_in.plot(x_display, intensity_map[0, :], 'b-', lw=2)
    
    # Add filled boxes for input waveguides (always displayed)
    input_colors = plt.cm.get_cmap('Set3', N)
    for i, p in enumerate(input_pos):
        ax_prof_in.axvspan((p - Sin_computed/2)*1e6, (p + Sin_computed/2)*1e6, 
                              alpha=0.15, color=input_colors(i), label=f'Input {i+1}' if i < 3 else '')
    
    # Mark MMI core boundaries (centered at x=0)
    ax_prof_in.axvline(x=-W/2*1e6, color='red', linestyle='--', alpha=0.5, label='MMI Core [-W/2, W/2]')
    ax_prof_in.axvline(x=W/2*1e6, color='red', linestyle='--', alpha=0.5)
    ax_prof_in.axvline(x=0, color='cyan', linestyle='-', lw=2, alpha=0.7, label='Center (x=0)')
    
    # Mark input positions
    for p in input_pos:
        ax_prof_in.axvline(x=p*1e6, color='k', linestyle=':', alpha=0.5)
    
    ax_prof_in.set_xlim(-W*1e6, W*1e6)  # Full 2W window centered
    ax_prof_in.set_xlabel('x [um]')
    ax_prof_in.set_ylabel('Intensity')
    ax_prof_in.set_title('Input Profile (x) at z=0')
    ax_prof_in.legend(loc='upper right', fontsize=8)

    # 3. Output Profile (Row 3) - CENTERED COORDINATES
    ax_prof = fig.add_subplot(gs[2, :])
    # x_grid is already in centered coordinates [-W, W]
    x_display = x_grid * 1e6
    ax_prof.plot(x_display, intensity_map[-1, :], 'b-', lw=2)
    
    # Add filled boxes for output waveguides (always displayed)
    output_colors = plt.cm.get_cmap('Set2', M)
    for j, p in enumerate(output_pos):
        ax_prof.axvspan((p - Sout_computed/2)*1e6, (p + Sout_computed/2)*1e6, 
                           alpha=0.15, color=output_colors(j), label=f'Output {j+1}' if j < 3 else '')
    
    # Mark MMI core boundaries (centered at x=0)
    ax_prof.axvline(x=-W/2*1e6, color='red', linestyle='--', alpha=0.5, label='MMI Core [-W/2, W/2]')
    ax_prof.axvline(x=W/2*1e6, color='red', linestyle='--', alpha=0.5)
    ax_prof.axvline(x=0, color='cyan', linestyle='-', lw=2, alpha=0.7, label='Center (x=0)')
    
    # Mark output positions
    for p in output_pos:
        ax_prof.axvline(x=p*1e6, color='k', linestyle=':', alpha=0.5)
    
    ax_prof.set_xlim(-W*1e6, W*1e6)  # Full 2W window centered
    ax_prof.set_xlabel('x [um]')
    ax_prof.set_ylabel('Intensity')
    ax_prof.set_title('Output Profile (x) at z=L')
    ax_prof.legend(loc='upper right', fontsize=8)

    # 4. Polar Plots (Row 4)
    colors = plt.cm.get_cmap('hsv', N+1)
    max_val = np.max(np.abs(phasors[-1, :, :]))
    limit = max_val * 1.1 if max_val > 1e-6 else 1.0

    for j in range(M):
        ax_p = fig.add_subplot(gs[3, j], projection='polar')
        ax_p.set_title(f'Out {j+1}')
        ax_p.set_ylim(0, limit)
        
        # Contributions
        for i in range(N):
            val = phasors[-1, j, i]
            ax_p.plot([0, np.angle(val)], [0, np.abs(val)], color=colors(i), lw=2, label=f'In {i+1}')
        
        # Total
        tot = np.sum(phasors[-1, j, :])
        ax_p.plot([0, np.angle(tot)], [0, np.abs(tot)], 'k--', lw=2, label='Total')
        
        if j == M-1:
            ax_p.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

    # 5. Z-Profile Plot (Row 5) - Grouped
    ax_z = fig.add_subplot(gs[4, :])
    ax_z.set_title('Z-Profile All Outputs')
    
    # Determine colors for Z-curves
    z_colors = plt.cm.get_cmap('tab10', M)
    
    # Max intensity for scaling
    max_int_z = np.max(intensity_map)*1.1
    
    for j in range(M):
        # Find x index for this output
        x_out = output_pos[j]
        ix = np.argmin(np.abs(x_grid - x_out))
        
        # Extract I(z) at this x
        I_z = intensity_map[:, ix]
        
        ax_z.plot(z_grid*1e6, I_z, color=z_colors(j), lw=1.5, label=f'Out {j+1}')
    
    # Vertical line for current L (end)
    ax_z.axvline(x=L_sim*1e6, color='r', linestyle='--', lw=1.0)
    
    ax_z.set_xlabel('z [um]')
    ax_z.set_xlim(0, L_sim*1e6)
    ax_z.set_ylim(0, max_int_z)
    ax_z.legend(loc='upper right', fontsize=8)

    # 6. Integrated Power in Core (Row 6) - Power evolution along z
    ax_power = fig.add_subplot(gs[5, :])
    ax_power.set_title('Integrated Power in Core [-W/2, W/2] vs. Propagation')
    
    # Calculate integrated intensity within core at each z (CENTERED COORDINATES)
    mask_core = (x_grid >= -W/2) & (x_grid <= W/2)
    dx = x_grid[1] - x_grid[0]
    
    # Compute power in core at each z step
    power_in_core_z = np.array([
        np.sum(intensity_map[iz, mask_core]) * dx for iz in range(len(z_grid))
    ])
    
    # Plot power evolution
    ax_power.plot(z_grid*1e6, power_in_core_z, 'darkblue', lw=2.5, label='Power in Core')
    ax_power.axhline(y=1.0, color='green', linestyle='--', lw=2, alpha=0.7, label='Input Reference (1.0)')
    
    # Vertical line for current L (end)
    ax_power.axvline(x=L_sim*1e6, color='r', linestyle='--', lw=1.0, alpha=0.7)
    
    ax_power.set_xlabel('z [um]')
    ax_power.set_ylabel('Integrated Power')
    ax_power.set_xlim(0, L_sim*1e6)
    ax_power.set_ylim(0, min(2.0, max(power_in_core_z)*1.2))  # Scale to see detail
    ax_power.grid(True, alpha=0.3)
    ax_power.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def simulate_contributions(N=2, M=2, L=None, W=10.0e-6 , n_core=2.0458, delta_n=0.0958, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=None, z_resolution=None, output_file=None, verbose=False, Din=None, Dout=None, Sin=None, Sout=None):
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
    n_core : float, default=2.0458
        Refractive index of the MMI core.
    delta_n : float, default=0.0958
        Index contrast (n_core - n_clad).
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
    Sin : float, optional
        Input waveguide mode width [m]. See :func:`simulate`.
    Sout : float, optional
        Output waveguide mode width [m]. See :func:`simulate`.

    Returns
    -------
    np.ndarray
        Complex amplitudes at the M outputs.
    """
    # Calculate n_clad from delta_n
    n_clad = n_core - delta_n
    
    # 1. Calculate Data
    data = compute_contributions(N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din=Din, Dout=Dout, Sin=Sin, Sout=Sout)
    
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
    output_amplitudes = simulate(N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, output_file=None, verbose=verbose, Din=Din, Dout=Dout)

    if verbose:
        print(f"Output amplitudes: {output_amplitudes}")

    return output_amplitudes
