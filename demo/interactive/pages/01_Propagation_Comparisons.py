import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pathlib import Path
import sys
import os

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
# Import utils from sibling directory
UTILS = Path(__file__).parent.parent / "utils"
if str(UTILS.parent) not in sys.path:
    sys.path.insert(0, str(UTILS.parent))

from utils.display import display_code
import helios
from helios import Wavefront

# --- Page Config ---
st.set_page_config(
    page_title="Propagation Comparisons",
    page_icon="🔦",
    layout="wide"
)

st.title("Propagation Comparisons 🔦")
st.markdown("""
This example compares different optical propagation methods available in HELIOS.
It simulates a simple optical system (pupil + optional lens) and propagates the wavefront
using various algorithms (Fraunhofer, Fresnel, ASM) to compare their results and validity domains.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "01_propagation_comparisons.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Parameters", expanded=True):
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        wavelength_um = st.number_input("Wavelength (µm)", value=0.633, format="%.4f")
        size_mm = st.number_input("Pupil Size (mm)", value=2.0)
    with col_p2:
        npix = st.selectbox("Resolution (px)", [128, 256, 512, 1024], index=1)
        focal_length_mm = st.number_input("Focal Length (mm)", value=50.0)

    wavelength = wavelength_um * u.um
    size = size_mm * u.mm
    focal_length = focal_length_mm * u.mm

    st.subheader("Methods to Test")
    available_methods = [
        'Fraunhofer', 'Fresnel', 'ASM', 'SCASM', 
        'Poppy', 'HCIPy', 'LightPipes', 
        'dLux_ASM', 'dLux_MFT', 'dLux_FFT'
    ]
    selected_methods = st.multiselect(
        "Select Propagation Methods", 
        available_methods,
        default=['Fraunhofer', 'Fresnel', 'ASM']
    )

run_btn = st.button("Run Simulation", type="primary")

if run_btn:
    with st.spinner("Simulating..."):
        # Define scenarios
        scenarios = [
            {"name": "Focal Plane (z=f)", "distance": focal_length, "use_lens": True},
            {"name": "Near Field (z=f/2)", "distance": focal_length / 2, "use_lens": True},
            # {"name": "Far Field (z=10f)", "distance": focal_length * 10, "use_lens": True}, # Can take too long
            {"name": "Free Space (No Lens)", "distance": 10*size, "use_lens": False},
        ]
        
        # Initial Wavefront
        wf_in = Wavefront(wavelength=wavelength, size=size, npix=npix)
        y, x = wf_in.coordinates()
        r = np.sqrt(x**2 + y**2)
        mask = r <= (size / 2)
        wf_in[:] = mask.astype(complex)
        
        a = size / 2

        # Loop over scenarios
        for scenario in scenarios:
            z = scenario["distance"]
            use_lens = scenario["use_lens"]
            name = scenario["name"]
            
            st.subheader(f"Scenario: {name}")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**Distance z:** {z}")
            with col2:
                N_F = (a**2 / (z * wavelength)).decompose()
                st.write(f"**Fresnel Number:** {N_F:.2f}")

            # Plotting
            n_methods = len(selected_methods) + 1 # +1 for Input
            cols = 4
            rows = (n_methods + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
            axes = np.atleast_1d(axes).flatten()
            
            # Input
            ax_in = axes[0]
            img_in = wf_in.intensity
            extent_in, xl_in, yl_in = helios.core.wavefront.get_smart_extent(wf_in.shape, wf_in.pixel_scale)
            ax_in.imshow(img_in, extent=extent_in, cmap='gray', origin='lower')
            ax_in.set_title("Input")
            ax_in.set_xlabel(xl_in)
            ax_in.set_ylabel(yl_in)
            
            for i, method in enumerate(selected_methods):
                ax = axes[i+1]
                try:
                     # Reset wavefront
                    wf = wf_in.copy()
                    
                    f_arg = focal_length if use_lens else None
                    output_npix = npix # Keep same resolution
                    
                    wf_out = wf.propagate(
                        distance=z,
                        focal_length=f_arg,
                        output_npix=output_npix,
                        regime=method
                    )
                    
                    # Visualization
                    img = wf_out.intensity
                    img_log = np.log10(img + 1e-12)
                    
                    extent, xl, yl = helios.core.wavefront.get_smart_extent(wf_out.shape, wf_out.pixel_scale)
                    
                    im = ax.imshow(img_log, extent=extent, cmap='inferno', origin='lower')
                    ax.set_title(f"{method}")
                    ax.set_xlabel(xl)
                    
                    # Energy
                    ratio = wf_out.integrated_intensity / wf_in.integrated_intensity
                    ax.text(0.05, 0.95, f"E: {ratio:.2f}", transform=ax.transAxes, color='white', fontsize=8, va='top')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, "Error", ha='center', va='center', color='red', transform=ax.transAxes)
                    ax.set_title(f"{method} (Failed)")
                    st.toast(f"{method} failed: {e}", icon="⚠️")

            # Hide unused
            for j in range(n_methods, len(axes)):
                axes[j].axis('off')
            
            st.pyplot(fig)
