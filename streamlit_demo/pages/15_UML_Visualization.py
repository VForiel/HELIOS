import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from copy import deepcopy as copy
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
import helios.components.photonics as photonics
import helios.components.photonics.fibers as fibers

# --- Page Config ---
st.set_page_config(
    page_title="UML Visualization",
    page_icon="📊",
    layout="wide"
)

st.title("UML Visualization 📊")
st.markdown("""
Generate automated **UML Block Diagrams** of your optical pipeline.
""")

# --- Show Code ---
EXAMPLE_PATH = ROOT / "examples" / "15_uml_visualization.py"
display_code(EXAMPLE_PATH)

st.divider()

# --- Interactive Demo ---

chart_type = st.radio("Select Diagram", ["Exoplanet Detection System", "Interferometric Beam Combiner"])

if st.button("Generate Diagram", type="primary"):
    
    with st.spinner("Generating Diagram..."):
        fig = None
        
        if chart_type == "Exoplanet Detection System":
            exo_scene = helios.Scene(distance=10*u.pc, name="Scene")
            exo_scene.add(helios.Star(temperature=5700*u.K, magnitude=5, position=(0, 0)))
            exo_scene.add(helios.Planet(temperature=300*u.K, magnitude=22, position=(100*u.mas, 0*u.mas)))

            atmosphere = helios.Atmosphere(rms=200*u.nm, wind_speed=8*u.m/u.s, seed=42, name="Atmosphere")
            elt = helios.TelescopeArray(pupil=helios.Pupil.elt(), size=39*u.m, name="ELT")
            elt.add_position(0, 0)
            ao = helios.AdaptiveOptics(coeffs={(1, 1): 0.15, (2, 0): 0.08}, name="AO")
            coronagraph = helios.Coronagraph(phase_mask='4quadrants', name="Coronagraph")
            bs = helios.BeamSplitter(cutoff=0.5, name="Beam Splitter")
            camera1 = helios.Camera(pixels=(512, 512), name="Cam 1")
            camera2 = helios.Camera(pixels=(256, 256), name="Cam 2")

            exo_ctx = helios.Context()
            exo_ctx.add_layer(exo_scene)
            exo_ctx.add_layer(atmosphere)
            exo_ctx.add_layer(elt)
            exo_ctx.add_layer(ao)
            exo_ctx.add_layer(coronagraph)
            exo_ctx.add_layer(bs)
            exo_ctx.add_layer([camera1, camera2])
            
            # Using plot_uml_diagram to return figure if supported, or we capture it
            # The example uses return_type='image' which returns an array
            img_arr = exo_ctx.plot_uml_diagram(return_type='image', figsize=(18, 8))
            
            fig = plt.figure(figsize=(18, 8))
            plt.imshow(img_arr)
            plt.axis('off')

        else: # Beam Combiner
            ctx = helios.Context()
            scene = helios.Scene(distance=10*u.pc, name="Target System")
            scene.add(helios.Star(temperature=5700*u.K, magnitude=5))
            scene.add(helios.Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU)))
            atmosphere = helios.Atmosphere(name="Atmosphere")
            
            pupil = helios.Pupil(8*u.m)
            telescopes = helios.TelescopeArray(pupil=pupil, size=8*u.m, name="Interferometer Array")
            telescopes.add_position(-10, -10)
            telescopes.add_position(10, -10)
            telescopes.add_position(10, 10)
            telescopes.add_position(-10, 10)

            ao_system = helios.AdaptiveOptics(name="AO System")
            fiber_in = fibers.FiberIn(modes=1, name="Fiber Injection")
            tops = photonics.TOPS(phase=0.0, name="Phase Shifter")
            
            mmi_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            mmi = photonics.MMI(matrix=mmi_matrix, name="Beam Combiner (MMI)")
            
            cross_mmi_matrix = np.array([
                    [np.exp(1j*np.pi/4), np.exp(-1j*np.pi/4)],
                    [np.exp(-1j*np.pi/4), np.exp(1j*np.pi/4)]
                ]) / np.sqrt(2)
            cross_mmi = photonics.MMI(matrix=cross_mmi_matrix, name="Cross MMI")
            
            fiber_out = fibers.FiberOut(name="Detector Port")
            cam = helios.Camera(pixels=(1,1), name="Photodiode")
            
            swap1 = photonics.Swap(mapping=[0, 2, 1, 3], name="Router")
            swap2 = photonics.Swap(mapping=[0, 1, 3, 2, 5, 4, 6], name="Router")
            y_splitter = photonics.YSplitter(name="Splitter")

            ctx.add_layer(scene)
            ctx.add_layer(atmosphere)
            ctx.add_layer(telescopes)
            ctx.add_layer([copy(ao_system) for _ in range(4)])
            ctx.add_layer([copy(fiber_in) for _ in range(4)])
            ctx.add_layer([copy(tops) for _ in range(4)])
            ctx.add_layer([copy(mmi) for _ in range(2)])
            ctx.add_layer(swap1)
            ctx.add_layer([copy(tops) for _ in range(4)])
            ctx.add_layer([copy(mmi) for _ in range(2)])
            ctx.add_layer([None] + [copy(y_splitter) for _ in range(3)])
            ctx.add_layer(swap2)
            ctx.add_layer([None] + [copy(tops) for _ in range(6)])
            ctx.add_layer([None] + [copy(cross_mmi) for _ in range(3)])
            ctx.add_layer([copy(fiber_out) for _ in range(7)])
            ctx.add_layer([copy(cam) for _ in range(7)])

            chip = photonics.PhotonicChip(inputs=2, lambda0=1.55*u.um, name="Beam Combiner Chip")
            for elem in [fiber_in, tops, mmi, fiber_out, fiber_out]:
                elem.layer = chip
            
            # This returns the figure directly if no return_type is specified?
            # Example: fig = ctx.plot_uml_diagram(figsize=(20, 12), layer_spacing=2.5)
            # The example code assigns to `fig`.
            fig = ctx.plot_uml_diagram(figsize=(20, 12), layer_spacing=2.5)
            
        st.pyplot(fig)
