import streamlit as st
import matplotlib.pyplot as plt
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
    page_title="Scene Geometry",
    page_icon="🌌",
    layout="wide"
)

st.title("Scene Geometry 🌌")
st.markdown("""
This example demonstrates how to define a scene with astronomical objects and visualize their spatial distribution.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "02_scene_geometry.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Parameters", expanded=True):
    st.subheader("Scene Parameters")
    distance_pc = st.number_input("System Distance (pc)", value=10.0, min_value=1.0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Star")
        temp_k = st.number_input("Temperature (K)", value=5700.0)
        mag = st.number_input("Magnitude", value=5.0)
    with col2:
        st.subheader("Planet")
        planet_dist_au = st.number_input("Separation (AU)", value=1.0)
        planet_mass_jup = st.number_input("Mass (Jupiter Mass)", value=1.0)

run_btn = st.button("Generate Scene", type="primary")

if run_btn:
    # Create scene
    scene = helios.Scene(distance=distance_pc * u.pc)
    
    # Add objects
    star = helios.Star(
        temperature=temp_k * u.K, 
        magnitude=mag, 
        mass=1 * u.M_sun, 
        position=(0 * u.AU, 0 * u.AU)
    )
    planet = helios.Planet(
        mass=planet_mass_jup * u.M_jup, 
        position=(planet_dist_au * u.AU, 0 * u.AU)
    )
    zodi = helios.Zodiacal(brightness=0.5)
    exozodi = helios.ExoZodiacal(brightness=0.3)
    
    scene.add(star)
    scene.add(planet)
    scene.add(zodi)
    scene.add(exozodi)

    # Visualize
    st.write("### Scene Visualization")
    
    # Capture plot
    # Ideally scene.plot() should take an ax argument, if not we use global state
    # Checking source code of scene.plot() isn't possible here easily without checking files,
    # but standard matplotlib practice allows us to create a figure first.
    fig = plt.figure(figsize=(8, 8))
    scene.plot() # Assuming it plots to current figure
    st.pyplot(fig)
    plt.close(fig)
