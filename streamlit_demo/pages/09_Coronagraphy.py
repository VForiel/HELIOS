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
    page_title="Coronagraphy",
    page_icon="🌑",
    layout="wide"
)

st.title("Coronagraphy 🌑")
st.markdown("""
Demonstrates starlight suppression using coronagraphic phase masks.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "09_coronagraphy.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

with st.expander("Parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        planet_sep_au = st.slider("Planet Separation (AU)", 1.0, 10.0, 4.0)
    with col2:
        planet_mass_jup = st.number_input("Planet Mass (Jup)", value=10.0)

run_btn = st.button("Run Simulation", type="primary")

if run_btn:
    with st.spinner("Simulating..."):
        # Setup Scene
        scene = helios.Scene(distance=10*u.pc)
        star = helios.Star(temperature=5700*u.K, magnitude=5, mass=1*u.M_sun, position=(0*u.AU, 0*u.AU))
        planet = helios.Planet(mass=planet_mass_jup*u.M_jup, position=(planet_sep_au*u.AU, 0*u.AU))
        scene.add(star)
        scene.add(planet)

        lam = 550e-9 * u.m
        D = 6.5 * u.m
        fov = 1 * u.arcsec

        flux_ratio = planet.flux_at(lam)/star.flux_at(lam)
        st.write(f"**Planet/Star Flux Ratio:** {flux_ratio:.2e}")

        # Coronagraphs
        coro_vortex = helios.Coronagraph(phase_mask='vortex')
        coro_4q = helios.Coronagraph(phase_mask='4quadrants')

        # Render Scene
        scene_img, x, y = scene.render(npix=256, fov=fov, return_coords=True)
        extent = [x[0].value, x[-1].value, y[0].value, y[-1].value] # Might need unit check
        # Helper to ensure values are float
        extent = [float(v) for v in extent]

        # Apply Coronagraphs
        img_vortex = coro_vortex.image_from_scene(scene_img, soft=True, oversample=2, 
                                                  normalize=False, lam=lam, diameter=D, fov=fov)
        img_4q = coro_4q.image_from_scene(scene_img, soft=True, oversample=2, 
                                          normalize=False, lam=lam, diameter=D, fov=fov)

        # Calculate suppression
        # Avoid div by zero if max is 0 (unlikely)
        suppression_vortex = scene_img.max() / (img_vortex.max() + 1e-20)
        suppression_4q = scene_img.max() / (img_4q.max() + 1e-20)
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original
        im0 = axes[0].imshow(scene_img, origin='lower', cmap='gray', extent=extent)
        axes[0].set_title('Original Scene')
        plt.colorbar(im0, ax=axes[0], label='Intensity', fraction=0.046, pad=0.04)

        # Vortex
        im1 = axes[1].imshow(img_vortex, origin='lower', cmap='inferno', extent=extent)
        axes[1].set_title(f'Vortex\n(Suppr: {suppression_vortex:.1e}x)')
        plt.colorbar(im1, ax=axes[1], label='Intensity', fraction=0.046, pad=0.04)

        # 4Q
        im2 = axes[2].imshow(img_4q, origin='lower', cmap='inferno', extent=extent)
        axes[2].set_title(f'4-Quadrant\n(Suppr: {suppression_4q:.1e}x)')
        plt.colorbar(im2, ax=axes[2], label='Intensity', fraction=0.046, pad=0.04)

        plt.tight_layout()
        st.pyplot(fig)
