import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    page_title="Planetary System Geometry",
    page_icon="🌌",
    layout="wide"
)

st.title("Planetary System Geometry 🌌")
st.markdown("""
Define the spatial configuration of a planetary system. 
Add planets, adjust their positions, and toggle dust components to visualize the system geometry.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "demo" / "scripts" / "02_planetary_system_geometry.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

col_params, col_viz = st.columns([1, 1.5])

with col_params:
    st.subheader("Configuration")
    
    # System Distance (needed for angular conversion)
    distance_pc = st.number_input("System Distance (pc)", value=10.0, min_value=1.0, help="Distance to the system, used to convert AU to arcseconds.")

    st.write("#### Star")
    # Minimal star config (mass influences marker size in plot)
    star_mass = st.number_input("Star Mass (Sun Mass)", value=1.0, step=0.1)

    st.write("#### Planets")
    # Data editor for planets
    df_planets = pd.DataFrame([
        {"Name": "Planet b", "Sep (AU)": 1.0, "Angle (deg)": 0.0, "Mass (M_jup)": 1.0},
        {"Name": "Planet c", "Sep (AU)": 2.5, "Angle (deg)": 90.0, "Mass (M_jup)": 2.0},
    ])
    
    edited_planets = st.data_editor(
        df_planets,
        num_rows="dynamic",
        column_config={
            "Name": st.column_config.TextColumn("Name"),
            "Sep (AU)": st.column_config.NumberColumn("Separation (AU)", min_value=0.0, step=0.1),
            "Angle (deg)": st.column_config.NumberColumn("Angle (°)", min_value=0.0, max_value=360.0, step=10.0),
            "Mass (M_jup)": st.column_config.NumberColumn("Mass (M_jup)", min_value=0.0, step=0.1),
        },
        use_container_width=True
    )

    st.write("#### Dust")
    show_zodi = st.toggle("Zodiacal Dust (Local)", value=False)
    show_exozodi = st.toggle("Exozodiacal Dust (System)", value=True)

    run_btn = st.button("Generate Geometry", type="primary")

with col_viz:
    if run_btn:
        st.subheader("Visualization")
        
        # Create system
        system = helios.PlanetarySystem(distance=distance_pc * u.pc)
        
        # Add Star
        star = helios.Star(mass=star_mass * u.M_sun, position=(0 * u.AU, 0 * u.AU))
        system.add(star)
        
        # Add Planets
        for _, row in edited_planets.iterrows():
            sep = row["Sep (AU)"]
            angle_deg = row["Angle (deg)"]
            mass = row["Mass (M_jup)"]
            name = row["Name"]
            
            # Convert polar to cartesian
            angle_rad = np.deg2rad(angle_deg)
            x_au = sep * np.cos(angle_rad)
            y_au = sep * np.sin(angle_rad)
            
            planet = helios.Planet(
                mass=mass * u.M_jup,
                position=(x_au * u.AU, y_au * u.AU),
                name=name
            )
            system.add(planet)
            
        # Add Dust
        if show_zodi:
            system.add(helios.Zodiacal())
        if show_exozodi:
            system.add(helios.ExoZodiacal())
            
        # Plot
        # scene.plot() returns (fig, ax)
        fig, ax = system.plot()
        st.pyplot(fig)
        plt.close(fig)
