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

# --- Page Config ---
st.set_page_config(
    page_title="Chromatic PSF",
    page_icon="🌈",
    layout="wide"
)

st.title("Chromatic PSF 🌈")
st.markdown("""
Visualizes the degradation of the Point Spread Function (PSF) due to atmospheric turbulence 
at different wavelengths (chromatic effects) and optical path differences (OPD).
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "10_chromatic_psf.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        wl_nm = st.slider("Wavelength (nm)", 400, 2000, 550, step=50)
    with col2:
        opd_nm = st.slider("Atmospheric OPD RMS (nm)", 0, 500, 50, step=10)

run_btn = st.button("Simulate PSF", type="primary")

if run_btn:
    N = 256 # Reduced for speed
    
    # Pupil
    try:
        pupil = helios.Pupil.like('JWST')
    except:
        pupil = helios.Pupil(6.5*u.m)
        pupil.add_disk(6.5/2*u.m)

    p_amp = pupil.get_array(npix=N, soft=True)
    
    wavelength = wl_nm * 1e-9 * u.m
    
    # Ideal PSF
    wf_ideal = helios.Wavefront(wavelength=wavelength, npix=N)
    wf_ideal.field = p_amp.astype(np.complex128)
    field_ideal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf_ideal.field)))
    psf_ideal = np.abs(field_ideal) ** 2
    peak_ideal = psf_ideal.max()
    
    # Degraded PSF
    wf = helios.Wavefront(wavelength=wavelength, npix=N)
    wf.field = p_amp.astype(np.complex128)
    
    atm = helios.Atmosphere(rms=opd_nm * u.nm, seed=42)
    wf_atm = atm.process(wf)
    
    field_final = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf_atm.field)))
    psf_final = np.abs(field_final) ** 2
    
    # Normalize
    psf_norm_final = psf_final / peak_ideal
    psf_norm_ideal = psf_ideal / peak_ideal
    
    strehl = psf_final.max() / peak_ideal
    phase_rms_rad = 2 * np.pi * (opd_nm * 1e-9) / (wl_nm * 1e-9)
    
    # Visualization
    st.write(f"**Strehl Ratio:** {strehl:.3f}")
    st.write(f"**Phase RMS:** {phase_rms_rad:.2f} rad")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ideal
    im0 = axes[0].imshow(np.log10(psf_norm_ideal + 1e-10), origin='lower', cmap='inferno', vmin=-5, vmax=0)
    axes[0].set_title("Ideal PSF (Log)")
    axes[0].axis('off')
    
    # Degraded
    im1 = axes[1].imshow(np.log10(psf_norm_final + 1e-10), origin='lower', cmap='inferno', vmin=-5, vmax=0)
    axes[1].set_title(f"Degraded PSF (Log)\nOPD={opd_nm}nm")
    axes[1].axis('off')
    
    # Difference
    diff = psf_norm_ideal - psf_norm_final
    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title("Difference (Ideal - Degraded)")
    axes[2].axis('off')
    
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    st.pyplot(fig)
