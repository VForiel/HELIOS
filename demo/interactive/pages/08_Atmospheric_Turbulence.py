import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from pathlib import Path
import sys
import os

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
    page_title="Atmospheric Turbulence",
    page_icon="💨",
    layout="wide"
)

st.title("Atmospheric Turbulence 💨")
st.markdown("""
Simulate atmospheric turbulence effects using the Frozen-Flow hypothesis.
Values of phase are wrapped between -π and π.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "08_atmospheric_turbulence.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        rms_nm = st.number_input("RMS Error (nm)", value=100.0, min_value=0.0)
    with col2:
        wind_speed_ms = st.number_input("Wind Speed (m/s)", value=10.0, min_value=0.0)
    with col3:
        wind_dir_deg = st.slider("Wind Direction (deg)", 0, 360, 45)
    
    col4, col5 = st.columns(2)
    with col4:
        duration_s = st.number_input("Duration (s)", value=2.0, min_value=0.1, max_value=10.0)
    with col5:
        fps = st.slider("FPS", 5, 30, 10)

run_btn = st.button("Generate Animation", type="primary")

if run_btn:
    with st.spinner("Generating animation..."):
        # Create Atmosphere
        atm_flow = helios.Atmosphere(
            rms=rms_nm*u.nm, 
            wind_speed=wind_speed_ms*u.m/u.s, 
            wind_direction=wind_dir_deg, 
            seed=123
        )
        
        # Output path
        output_dir = ROOT / "demo" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "turbulence.gif"
        
        # Generate Animation using built-in method
        # plot_screen_animation handles pipeline context internally
        anim = atm_flow.plot_screen_animation(
            duration=duration_s*u.s,
            fps=fps,
            npix=256,
            filename=str(save_path),
            figsize=(8, 8)
        )
        
        # Display
        st.success(f"Animation generated!")
        st.image(str(save_path), caption="Frozen-Flow Turbulence Evolution")
        
        # Clean up figure context to avoid OOM in loop
        plt.close('all')
