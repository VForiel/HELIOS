import streamlit as st
import numpy as np
import os
from pathlib import Path
import sys

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
from helios.sim.mmi import simulate_contributions

# --- Page Config ---
st.set_page_config(
    page_title="MMI Contributions",
    page_icon="🎞️",
    layout="wide"
)

st.title("MMI Contributions 🎞️")
st.markdown("""
Generate animations showing Multimode Interference (MMI) propagation contributions.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "13_mmi.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

demo_choice = st.radio("Choose Simulation", ["2x2 Bracewell Nuller", "4x4 Kernel Nuller"])

if st.button("Generate & View Animation", type="primary"):
    
    # Define output path
    output_filename = f"mmi_demo_{demo_choice.replace(' ', '_')}.mp4"
    # Save to temp or local dir? Local is fine for demo
    output_path = ROOT / "generated" / "streamlit" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    params = {}
    
    with st.spinner("Running simulation and generating video... (This may take a minute)"):
        try:
            if demo_choice == "2x2 Bracewell Nuller":
                params = dict(
                    N=2, M=2, L=100e-6, W=10.0e-6,
                    Din=5.0e-6, Dout=5.0e-6, Sin=2.5e-6, Sout=2.5e-6,
                    n_core=2.0458, delta_n=0.0958, wavelength=1.55e-6,
                    input_amplitudes=np.sqrt(0.5) * np.array([1, 1j], dtype=complex),
                    num_modes=50, z_resolution=1.0e-6,
                    output_file=str(output_path),
                    verbose=False # To avoid stdout spam
                )
            else:
                params = dict(
                    N=4, M=4, L=400e-6, W=20e-6,
                    Din=4.0e-6, Dout=4.0e-6, Sin=5.0e-6, Sout=5.0e-6,
                    n_core=2.0458, delta_n=0.0958, wavelength=1.55e-6,
                    input_amplitudes=np.sqrt(0.25) * np.array([1, 1j, 1, 1j], dtype=complex),
                    num_modes=50, z_resolution=1.0e-6,
                    output_file=str(output_path),
                    verbose=False
                )

            simulate_contributions(**params)
            
            st.success("Animation generated!")
            if output_path.exists():
                st.video(str(output_path))
            else:
                st.error("Output file not found.")

        except Exception as e:
            st.error(f"Error during simulation: {e}")
            st.info("Ensure ffmpeg is installed and accessible.")
