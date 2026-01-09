import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os
import sys
import warnings
from pathlib import Path

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
# Import utils
UTILS = Path(__file__).parent.parent / "utils"
if str(UTILS.parent) not in sys.path:
    sys.path.insert(0, str(UTILS.parent))

from utils.display import display_code
from helios.sim.mmi import (
    calibrate_input_phases_genetic,
    calibrate_n_core_and_phases,
    plot_mmi_interactive,
)
from helios.sim.lp_modes import suppress_lp_warnings

# --- Page Config ---
st.set_page_config(
    page_title="MMI Interactive",
    page_icon="🎛️",
    layout="wide"
)

st.title("MMI Interactive 🎛️")
st.markdown("""
Interactive exploration of Multimode Interference (MMI) couplers, including
genetic phase calibration and index optimization.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "14_mmi_streamlit.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Helper Functions (Ported) ---

def _um(value_um: float):
    """Convert microns to meters; treat 0 or negative as None (auto)."""
    if value_um is None or value_um <= 0:
        return None
    return value_um * 1e-6


def _normalize_complex(amplitudes: List[complex]) -> List[complex]:
    norm = sum(np.abs(a) ** 2 for a in amplitudes)
    if norm > 0:
        return [a / np.sqrt(norm) for a in amplitudes]
    return amplitudes


def _normalize_real(amplitudes: List[float]) -> List[float]:
    norm = sum(np.abs(a) ** 2 for a in amplitudes)
    if norm > 0:
        return [float(a / np.sqrt(norm)) for a in amplitudes]
    return amplitudes


def _load_doc(section: str) -> str:
    """Load a markdown doc section from docs/learn/mmi."""
    # Modified path logic to work from this page location
    # ROOT is d:\HELIOS
    base_dir = ROOT / "docs" / "learn" / "mmi"
    filename = {
        "Overview": "index.md",
        "Physical Principles": "physics.md",
        "Design Rules": "design.md",
        "Numerical Implementation": "numerics.md",
        "Usage": "usage.md",
        "Validation": "validation.md",
    }.get(section)
    if filename is None:
        return ""
    path = base_dir / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"⚠️ Documentation file not found: {filename}"
    except Exception as exc:
        return f"⚠️ Error loading documentation: {exc}"


def _collect_inputs(num_inputs: int):
    amps = []
    phases = []
    for i in range(num_inputs):
        amp_key = f"amp_{i}"
        phase_key = f"phase_{i}"
        if amp_key not in st.session_state:
            st.session_state[amp_key] = 1.0 if i == 0 else 1.0
        if phase_key not in st.session_state:
            st.session_state[phase_key] = 0.0

        col_amp, col_phase = st.columns(2)
        amp_val = col_amp.slider(
            f"Amplitude {i+1}", 0.0, 2.0, st.session_state[amp_key], 0.05, key=amp_key
        )
        phase_pi = col_phase.slider(
            f"Phase {i+1} (×π rad)", 0.0, 2.0, st.session_state[phase_key], 0.01, key=phase_key
        )
        amps.append(amp_val)
        phases.append(phase_pi * np.pi)

    complex_vec = [amp * np.exp(1j * phi) for amp, phi in zip(amps, phases)]
    return _normalize_complex(complex_vec), _normalize_real(amps), phases


def run_simulation(params, complex_inputs):
    st.info("Running MMI simulation…")
    try:
        fig = plot_mmi_interactive(
            N=params["N"],
            M=params["M"],
            L=params["L"],
            W=params["W"],
            n_core=params["n_core"],
            delta_n=params["delta_n"],
            wavelength=params["wavelength"],
            input_amplitudes=complex_inputs,
            num_modes=params["num_modes"],
            num_z_steps=500,
            z_resolution=params["W"] * 2,
            Din=params["Din"],
            Dout=params["Dout"],
            Sin=params["Sin"],
            Sout=params["Sout"],
            verbose=False,
        )
        st.pyplot(fig, width="stretch") # Streamlit deprecation: check use_container_width
        plt.close(fig)
    except Exception as exc:
        st.error(f"Simulation Error: {exc}")


def run_phase_calibration(params, magnitudes):
    st.info("Calibrating input phases…")
    try:
        with suppress_lp_warnings():
            result = calibrate_input_phases_genetic(
                N=params["N"],
                M=params["M"],
                L=params["L"],
                W=params["W"],
                n_core=params["n_core"],
                delta_n=params["delta_n"],
                wavelength=params["wavelength"],
                input_amplitudes=np.array(magnitudes, dtype=float),
                bright_output_idx=params["bright_idx"],
                num_modes=params["num_modes"],
                num_z_steps=30,
                z_resolution=None,
                Din=params["Din"],
                Dout=params["Dout"],
                Sin=params["Sin"],
                Sout=params["Sout"],
                beta=0.8,
                initial_step=np.pi / 2,
                epsilon=1e-3,
                verbose=False,
            )
    except Exception as exc:
        st.error(f"Calibration Error: {exc}")
        return None

    st.success(
        f"Phase calibration complete. Best metric (null/bright): {result['best_metric']:.3e}"
    )
    st.write(f"Best phases [rad]: {result['best_phases']}")

    # Plot convergence history
    if "metric" in result:
        metric_array = np.asarray(result["metric"], dtype=float)
        if len(metric_array) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.semilogy(metric_array, "o-", lw=2, markersize=6, color="steelblue")
            ax.axhline(y=result["best_metric"], color="red", linestyle="--", lw=2, label=f"Best: {result['best_metric']:.3e}")
            ax.set_xlabel("Iteration", fontsize=11)
            ax.set_ylabel("Null Depth Metric (null/bright)", fontsize=11)
            ax.set_title("Phase Calibration Convergence", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Metric array is empty; no convergence plot to display.")
    else:
        st.warning("Result does not contain a 'metric' history; no plot to display.")

    st.session_state["phase_updates"] = [float(phi / np.pi) for phi in result["best_phases"]]
    st.info("Calibrated phases are staged. Click 'Apply calibrated phases' to update sliders.")
    if st.button("Apply calibrated phases", key="apply_phases_btn"):
        st.rerun()

    return result


def run_ncore_calibration(params, magnitudes):
    st.info("Two-stage n_core + phase calibration…")

    progress_coarse = st.progress(0.0, text="Stage 1: coarse scan")
    status_coarse = st.empty()
    status_grad = st.empty()

    def cb_coarse(current, total):
        progress_coarse.progress(current / total)
        status_coarse.write(f"Stage 1: {current}/{total}")

    def cb_grad(iteration, delta_n):
        status_grad.write(f"Stage 2: iteration {iteration}, Δn_core = {delta_n:.4f}")

    try:
        with suppress_lp_warnings(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calibrate_n_core_and_phases(
                N=params["N"],
                M=params["M"],
                L=params["L"],
                W=params["W"],
                n_core_initial=params["n_core"],
                n_core_min=1.0,
                n_core_max=2.0 * params["n_core"],
                delta_n=params["delta_n"],
                wavelength=params["wavelength"],
                input_amplitudes=np.array(magnitudes, dtype=float),
                bright_output_idx=params["bright_idx"],
                num_modes=params["num_modes"],
                num_z_steps=30,
                z_resolution=None,
                Din=params["Din"],
                Dout=params["Dout"],
                Sin=params["Sin"],
                Sout=params["Sout"],
                n_core_steps_coarse=20,
                gradient_convergence_threshold=1e-3,
                gradient_initial_step=0.01,
                beta=0.8,
                initial_step=np.pi / 2,
                epsilon=1e-3,
                verbose=False,
                progress_callback_coarse=cb_coarse,
                progress_callback_gradient=cb_grad,
            )
    except Exception as exc:
        status_grad.write("Error during optimization")
        st.error(f"n_core calibration error: {exc}")
        return None

    progress_coarse.progress(1.0, text="Stage 1 complete")
    status_coarse.write("Stage 1 complete")
    status_grad.write(f"Stage 2 complete in {len(result['n_core_values_gradient'])-1} iterations")

    st.success(
        f"Optimal n_core = {result['best_n_core']:.4f}, best metric = {result['best_metric']:.3e}"
    )
    st.write(f"Best phases [rad]: {result['best_phases']}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Coarse scan
    if "n_core_values_coarse" in result and result["n_core_values_coarse"] is not None:
        ax_left = axes[0]
        ax_left.semilogy(
            result["n_core_values_coarse"],
            result["metrics_coarse"],
            "o-", lw=2, markersize=6, color="coral", label="Coarse Scan"
        )
        ax_left.axvline(x=result["best_n_core"], color="red", linestyle="--", lw=2, label=f"Best: {result['best_n_core']:.4f}")
        ax_left.set_xlabel("n_core", fontsize=11)
        ax_left.set_ylabel("Null Depth Metric (null/bright)", fontsize=11)
        ax_left.set_title("Stage 1: Coarse Grid Scan", fontsize=12, fontweight="bold")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(fontsize=10)
    
    # Right: Gradient descent refinement
    if "n_core_values_gradient" in result and result["n_core_values_gradient"] is not None:
        ax_right = axes[1]
        n_iter = len(result["n_core_values_gradient"])
        ax_right.semilogy(
            range(n_iter),
            result["metrics_gradient"],
            "s-", lw=2, markersize=6, color="mediumseagreen", label="Gradient Descent"
        )
        ax_right.axhline(y=result["best_metric"], color="red", linestyle="--", lw=2, label=f"Best: {result['best_metric']:.3e}")
        ax_right.set_xlabel("Iteration", fontsize=11)
        ax_right.set_ylabel("Null Depth Metric (null/bright)", fontsize=11)
        ax_right.set_title("Stage 2: Gradient Descent Refinement", fontsize=12, fontweight="bold")
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(fontsize=10)
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.session_state["n_core_override"] = float(result["best_n_core"])
    st.session_state["phase_updates"] = [float(phi / np.pi) for phi in result["best_phases"]]
    st.rerun()

    return result


# --- Main Logic ---

with st.expander("Geometry & Materials", expanded=True):
    st.header("Geometry & Materials")
    N = st.number_input("N inputs", 1, 8, 4)
    M = st.number_input("M outputs", 1, 8, 4)
    W_um = st.number_input("W (µm)", value=20.0, min_value=2.0, max_value=50.0, step=0.5)
    L_um = st.number_input("L (µm) — 0 = auto", value=440.0, min_value=0.0, step=10.0)
    Din_um = st.number_input("Din (µm) — 0 = auto", value=5.0, min_value=0.0, step=0.5)
    Dout_um = st.number_input("Dout (µm) — 0 = auto", value=5.0, min_value=0.0, step=0.5)
    Sin_um = st.number_input("Sin (µm) — 0 = auto", value=4.5, min_value=0.0, step=0.5)
    Sout_um = st.number_input("Sout (µm) — 0 = auto", value=4.5, min_value=0.0, step=0.5)

    st.header("Optics")
    wavelength_um = st.number_input("λ (µm)", value=1.55, min_value=0.4, max_value=4.0, step=0.01, format="%.2f")
    n_core_val = st.number_input("n_core", 1.0, 4.0, st.session_state.get("n_core_override", 2.0458), format="%.4f")
    delta_n_val = st.number_input("Δn (n_core − n_clad)", 0.001, 0.5, 0.0958, format="%.4f")
    num_modes = st.number_input("Num modes (upper bound)", value=200, min_value=10, max_value=200, step=1)
    bright_idx = st.number_input("Bright output index", 0, max(0, int(M) - 1), 0)

# Apply any pending phase updates before widgets are instantiated
if "phase_updates" in st.session_state:
    phases_to_apply = st.session_state.pop("phase_updates")
    for i, val in enumerate(phases_to_apply[: int(N)]):
        st.session_state[f"phase_{i}"] = float(val)

st.subheader("Input amplitudes & phases")
complex_inputs, magnitudes, phases = _collect_inputs(int(N))

params = dict(
    N=int(N),
    M=int(M),
    L=_um(L_um),
    W=_um(W_um),
    Din=_um(Din_um),
    Dout=_um(Dout_um),
    Sin=_um(Sin_um),
    Sout=_um(Sout_um),
    wavelength=_um(wavelength_um),
    n_core=float(st.session_state.get("n_core_override", n_core_val)),
    delta_n=float(delta_n_val),
    num_modes=int(num_modes),
    bright_idx=int(bright_idx),
)

st.write(
    "Use the buttons below to simulate, calibrate phases, or jointly optimize n_core + phases."
)

col_sim, col_calib, col_ncore = st.columns(3)
if col_sim.button("Simulate", use_container_width=True):
    run_simulation(params, complex_inputs)

if col_calib.button("Calibrate Phases", use_container_width=True):
    res = run_phase_calibration(params, magnitudes)
    if res is not None:
        st.info("Phase sliders updated to calibrated values. Re-run Simulate to visualize.")

if col_ncore.button("Determine n_core", use_container_width=True):
    res = run_ncore_calibration(params, magnitudes)
    if res is not None:
        st.info("n_core and phases updated. Re-run Simulate to visualize.")

st.markdown("---")
st.markdown(
    "**Tips:** Set L=0 to auto-pick from the self-imaging formula; set Din/Dout/Sin/Sout to 0 to use defaults."
)

st.markdown("---")
st.subheader("Documentation (learn/mmi)")
doc_section = st.selectbox(
    "Choose a section",
    [
        "Overview",
        "Physical Principles",
        "Design Rules",
        "Numerical Implementation",
        "Usage",
        "Validation",
    ],
)
doc_content = _load_doc(doc_section)
if doc_content:
    st.markdown(doc_content)
else:
    st.info("No documentation available for this section.")
