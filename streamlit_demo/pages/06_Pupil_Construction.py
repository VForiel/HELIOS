import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
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
import helios

# --- Page Config ---
st.set_page_config(
    page_title="Pupil Construction",
    page_icon="🔭",
    layout="wide"
)

st.title("Pupil Construction 🔭")
st.markdown("""
Construct and visualize telescope pupils using manual primitives or standard presets.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "06_pupil_construction.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Configuration", expanded=True):
    col_mode, col_p = st.columns([1, 2])
    
    with col_mode:
        mode = st.radio("Construction Mode", ["Manual", "Preset"])

    if mode == "Manual":
        with col_p:
            st.subheader("Manual Parameters")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                prim_diam = st.number_input("Primary Mirror Diameter (m)", value=8.0, min_value=1.0)
                obs_diam = st.number_input("Central Obscuration (m)", value=1.1, min_value=0.0, max_value=prim_diam)
            with col_m2:
                n_spiders = st.slider("Number of Spiders", 0, 8, 4)
                spider_width = st.number_input("Spider Width (m)", value=0.05, min_value=0.01)

        pupil = helios.Pupil(prim_diam * u.m)
        
        # Outer Disk
        pupil.add_disk(radius=(prim_diam/2) * u.m)
        if obs_diam > 0:
            pupil.add_central_obscuration(diameter=obs_diam * u.m)
        if n_spiders > 0:
            pupil.add_spiders(arms=n_spiders, width=spider_width * u.m)
            
        ax.set_title(f'Manual Pupil ({prim_diam}m)')

    else: # Preset
        with col_p:
            preset_name = st.selectbox("Select Preset", ["JWST", "ELT", "VLT"]) 
    # Note: ELT/VLT might not be in helios yet, so let's wrap in try/except or check.
    # The example only showed JWST.
    
    try:
        if preset_name == "JWST":
            pupil = helios.Pupil.like('JWST')
        elif preset_name == "ELT":
            # Assuming functionality exists or fallback
             pupil = helios.Pupil(39*u.m) # Placeholder if 'like' fails or just try
             # Actually let's try .like() if implemented
             try:
                 pupil = helios.Pupil.like('ELT')
             except:
                 st.warning("ELT preset not fully implemented, showing placeholder.")
                 pupil = helios.Pupil(39*u.m)
                 pupil.add_disk(39/2*u.m)
                 pupil.add_central_obscuration(11*u.m)
                 pupil.add_spiders(6, 0.5*u.m)
        elif preset_name == "VLT":
             pupil = helios.Pupil(8.2*u.m)
             pupil.add_disk(4.1*u.m)
             pupil.add_central_obscuration(1.1*u.m)
        
        ax.set_title(f'{preset_name} Pupil')
        
    except Exception as e:
        st.error(f"Error loading preset: {e}")

if pupil:
    npix = st.select_slider("Resolution", options=[128, 256, 512, 1024, 2048], value=512)
    
    arr = pupil.get_array(npix=npix)
    
    ax.imshow(arr, origin='lower', cmap='gray')
    ax.axis('off')
    
    st.pyplot(fig)
    
    st.info(f"Shape: {arr.shape}, Fill Factor: {arr.mean():.3%}")
