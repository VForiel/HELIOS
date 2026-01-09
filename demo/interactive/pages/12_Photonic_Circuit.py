import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from pathlib import Path
import sys

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
# Import utils
UTILS = Path(__file__).parent.parent / "utils"
if str(UTILS.parent) not in sys.path:
    sys.path.insert(0, str(UTILS.parent))

from utils.display import display_code
import helios
import helios.components.photonics as photonics
import helios.components.photonics.fibers as fibers
from helios.core.wavefront import Wavefront

# --- Page Config ---
st.set_page_config(
    page_title="Photonic Circuit",
    page_icon="🔌",
    layout="wide"
)

st.title("Photonic Circuit 🔌")
st.markdown("""
Simulates a simple photonic integrated circuit:
**FiberIn ⮕ Y-Splitter ⮕ Phase Shifters ⮕ MMI ⮕ FiberOuts**.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "12_photonic_circuit.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Parameters", expanded=True):
    phase_bot = st.slider("Bottom Phase Shift (rad)", 0.0, 2*np.pi, np.pi/2)

run_btn = st.button("Run Simulation", type="primary")

if run_btn:
    with st.spinner("Propagating..."):
        # Components
        fiber_in = fibers.FiberIn(modes=1)
        splitter = photonics.YSplitter()
        tops_top = photonics.TOPS(phase=0.0)
        tops_bot = photonics.TOPS(phase=phase_bot) # Interactive
        mmi_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        mmi = photonics.MMI(matrix=mmi_matrix)
        fiber_out_1 = fibers.FiberOut()
        fiber_out_2 = fibers.FiberOut()

        # Input Wavefront
        wf_in = Wavefront(wavelength=1.55*u.um, npix=128)
        x = np.linspace(-5, 5, 128)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        wf_in.field = np.exp(-R**2).astype(np.complex128)

        # Process
        wf_coupled = fiber_in.process(wf_in)
        wf_split = splitter.process(wf_coupled)
        wf_top = tops_top.process(wf_split[0])
        wf_bot = tops_bot.process(wf_split[1])
        wf_out_mmi = mmi.process([wf_top, wf_bot])
        output_1 = fiber_out_1.process(wf_out_mmi[0])
        output_2 = fiber_out_2.process(wf_out_mmi[1])

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Helper for dimensions
        def get_2d(w):
            val = w.value
            return val[0] if val.ndim == 3 else val

        wf_in_2d = get_2d(wf_in)
        out1_2d = get_2d(output_1)
        out2_2d = get_2d(output_2)
        
        # Metrics
        p1 = np.sum(np.abs(out1_2d)**2)
        p2 = np.sum(np.abs(out2_2d)**2)
        total = p1 + p2
        
        st.write(f"**Output Power Split:** Port 1: {p1/total:.1%} | Port 2: {p2/total:.1%}")
        
        axes[0].imshow(np.abs(wf_in_2d)**2, cmap='inferno')
        axes[0].set_title("Input Field Intensity")
        axes[0].axis('off')
        
        axes[1].imshow(np.abs(out1_2d)**2, cmap='inferno')
        axes[1].set_title("Output Port 1")
        axes[1].axis('off')
        
        axes[2].imshow(np.abs(out2_2d)**2, cmap='inferno')
        axes[2].set_title("Output Port 2")
        axes[2].axis('off')
        
        st.pyplot(fig)
