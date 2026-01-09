"""
HELIOS Streamlit Showcase
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Add src to path
ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(
    page_title="HELIOS Showcase",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("HELIOS: Hierarchical End-to-end Lightpath & Instrumental response Observational Simulation ☀️")

st.markdown(
    """
    Welcome to the **HELIOS** interactive showcase.
    
    This application demonstrates the capabilities of the **HELIOS** framework, 
    designed for simulating optical systems, specifically for:
    
    *   **Wavefront Propagation** (Fraunhofer, Fresnel, ASM)
    *   **Interferometry** (VLTI, LIFE constellations)
    *   **Photonic Integrated Circuits** (MMI, directional couplers)
    *   **Exoplanet Detection** (Nulling interferometry)

    ## How to use this app
    
    Select an example from the sidebar to explore specific features.
    Each page provides:
    1.  An **interactive demonstration** or static output of the simulation.
    2.  The **source code** used to generate the result, allowing you to learn how to use the library.
    """
)

st.info("👈 Select a demo from the sidebar to get started!")

st.markdown("---")
st.caption("HELIOS — VForiel")
