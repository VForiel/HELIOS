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
    page_title="Spectral Energy Distribution",
    page_icon="🌈",
    layout="wide"
)

st.title("Spectral Energy Distribution 🌈")
st.markdown("""
This example visualizes the Spectral Energy Distributions (SEDs) of astronomical objects.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "03_spectral_energy_distribution.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Parameters", expanded=True):
    st.subheader("Parameters")
    distance_pc = st.number_input("System Distance (pc)", value=10.0, min_value=1.0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Star")
        temp_k = st.number_input("Temperature (K)", value=5700.0)
        mag = st.number_input("Magnitude", value=5.0)
    
    with col2:
        st.subheader("Planet")
        planet_dist_au = st.number_input("Separation (AU)", value=1.0)
        planet_radius_jup = st.number_input("Radius (Jupiter Radius)", value=1.0)
        planet_albedo = st.slider("Albedo", 0.0, 1.0, 0.3)

run_btn = st.button("Plot SED", type="primary")

if run_btn:
    # Create objects
    scene = helios.Scene(distance=distance_pc * u.pc)
    star = helios.Star(
        temperature=temp_k * u.K, 
        magnitude=mag, 
        mass=1 * u.M_sun, 
        position=(0 * u.AU, 0 * u.AU)
    )
    planet = helios.Planet(
        mass=1 * u.M_jup, 
        position=(planet_dist_au * u.AU, 0 * u.AU), 
        albedo=planet_albedo, 
        radius=planet_radius_jup * u.R_jup
    )
    
    # Add to scene
    scene.add(star)
    scene.add(planet)

    # Plot SEDs
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Assuming plot_sed returns the axis or modifies it
    # If the API requires ax passed in:
    try:
        star.plot_sed(ax=ax, color='gold', label='Star')
        planet.plot_sed(ax=ax, color='blue', label='Planet')
    except TypeError:
         # Fallback if ax arg not supported (unlikely based on example)
        star.plot_sed()
        planet.plot_sed()

    ax.set_title(f'Spectral Energy Distributions (d={distance_pc}pc)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
