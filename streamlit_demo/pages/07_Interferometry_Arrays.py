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
    page_title="Interferometry Arrays",
    page_icon="✨",
    layout="wide"
)

st.title("Interferometry Arrays ✨")
st.markdown("""
Configure and visualize interferometric arrays like VLTI or LIFE.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "07_interferometry_arrays.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Configuration", expanded=True):
    array_type = st.selectbox("Array Type", ["VLTI (UTs)", "VLTI (ATs)", "LIFE"])

interferometer = None

if array_type == "VLTI (UTs)":
    interferometer = helios.TelescopeArray.vlti(uts=True)
elif array_type == "VLTI (ATs)":
    # Assuming helios supports ATs via arg or another method. 
    # Example only showed uts=True. Let's try uts=False.
    interferometer = helios.TelescopeArray.vlti(uts=False)
elif array_type == "LIFE":
    # Assuming life() factory exists
    try:
        interferometer = helios.TelescopeArray.life()
    except AttributeError:
        st.warning("'life' factory not found, using generic placeholder.")
        # Fallback manual construction just to show something?
        interferometer = helios.TelescopeArray(name="LIFE-Mock")
        interferometer.add_telescope(position=(10*u.m, 0*u.m), pupil=helios.Pupil(2*u.m))
        interferometer.add_telescope(position=(-10*u.m, 0*u.m), pupil=helios.Pupil(2*u.m))
        interferometer.add_telescope(position=(0*u.m, 17*u.m), pupil=helios.Pupil(2*u.m))
        interferometer.add_telescope(position=(0*u.m, -17*u.m), pupil=helios.Pupil(2*u.m))

col1, col2 = st.columns([1, 2])

with col1:
    st.write(f"**Name:** {interferometer.name}")
    st.write(f"**Telescopes:** {interferometer.num_telescopes}")
    
    with st.expander("Show Baselines"):
        st.write(interferometer.get_baseline_array())

with col2:
    show_pupils = st.checkbox("Show Pupils", value=True)
    scale = st.slider("Pupil Scale", 0.1, 5.0, 0.5)
    
    # Plot
    # plot_array likely uses plt.gca() or creates a new figure.
    # We'll create a figure to be safe.
    fig = plt.figure(figsize=(8, 8))
    interferometer.plot_array(show_pupils=show_pupils, pupil_scale=scale)
    st.pyplot(fig)
