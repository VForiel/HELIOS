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
    page_title="End-to-End Simulation",
    page_icon="🎬",
    layout="wide"
)

st.title("End-to-End Simulation 🎬")
st.markdown("""
Runs a full end-to-end simulation: **Scene ⮕ Telescope ⮕ Camera**.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "11_end_to_end_simulation.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Configuration", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scene")
        scene_dist = st.number_input("Distance (pc)", value=10.0)
        
        st.subheader("Planet")
        planet_sep = st.number_input("Separation (arcsec)", value=0.1)
        planet_mass = st.number_input("Mass (Jup)", value=1.0)
        
    with col2:
        st.subheader("Instrument")
        telescope_diam = st.number_input("Telescope Diameter (m)", value=2.0)
        wavelength_nm = st.number_input("Wavelength (nm)", value=600.0)

run_btn = st.button("Run Simulation", type="primary")

if run_btn:
    with st.spinner("Simulating..."):
        # 1. Scene
        scene = helios.Scene(distance=scene_dist*u.pc)
        star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.arcsec, 0*u.arcsec))
        # Planet off-axis
        planet = helios.Planet(mass=planet_mass*u.M_jup, position=(planet_sep*u.arcsec, 0*u.arcsec))
        scene.add(star)
        scene.add(planet)

        # 2. Collectors
        pupil_obs = helios.Pupil(diameter=telescope_diam*u.m)
        # Add simple obscuration
        pupil_obs.add_disk(center=(0*u.m, 0*u.m), radius=(telescope_diam/2)*u.m)
        collectors = helios.TelescopeArray(pupil=pupil_obs, size=telescope_diam*u.m, name="Simple Array")
        collectors.add_position(x=0*u.m, y=0*u.m) # Single aperture

        # 3. Camera
        camera = helios.Camera(pixels=(256, 256))

        # 4. Context & Simulation
        pipeline = helios.Pipeline(wavelength=wavelength_nm*u.nm, npix=512)
        pipeline.add_layer(scene)
        pipeline.add_layer(collectors)
        pipeline.add_layer(camera)

        result = pipeline.observe()
        
        # Display
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(result, origin='lower', cmap='inferno')
        plt.colorbar(im, label='Intensity')
        ax.set_title('Simulated Observation Result')
        st.pyplot(fig)
        
        st.info(f"Result statistics: Min={result.min():.2e}, Max={result.max():.2e}")
