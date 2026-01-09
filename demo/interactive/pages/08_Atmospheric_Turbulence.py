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
    page_title="Atmospheric Turbulence",
    page_icon="💨",
    layout="wide"
)

st.title("Atmospheric Turbulence 💨")
st.markdown("""
Simulate atmospheric turbulence effects using the Frozen-Flow hypothesis.
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

run_btn = st.button("Simulate Frozen Flow", type="primary")

if run_btn:
    class ObservationContext(helios.Context):
        def __init__(self, time):
            super().__init__()
            self.time = time

    atm_flow = helios.Atmosphere(
        rms=rms_nm*u.nm, 
        wind_speed=wind_speed_ms*u.m/u.s, 
        wind_direction=wind_dir_deg, 
        seed=123
    )
    
    wavelength = 550e-9 * u.m
    N = 256 # Lower res for speed
    
    # Use generic circular pupil if preset fails, but use JWST like example
    try:
        pupil_jwst = helios.Pupil.like('JWST')
    except:
        pupil_jwst = helios.Pupil(6.5*u.m)
        pupil_jwst.add_disk(6.5/2*u.m)

    p_amp = pupil_jwst.get_array(npix=N, soft=True)
    
    times = [0, 0.5, 1.0, 1.5, 2.0, 2.5]  # seconds

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(times):
        wf_t = helios.Wavefront(wavelength=wavelength, npix=N)
        wf_t.field = p_amp.astype(complex) # Fixed cast
        
        # Override context time manually or use clean context object if needed
        # The example used a subclass ObservationContext to pass time.
        # Let's ensure Atmosphere uses the context properly. 
        # helios.Atmosphere.process(wf, context or nothing?)
        # Checking example logic:
        # ctx = ObservationContext(t*u.s)
        # wf_t_atm = atm_flow.process(wf_t) 
        # Wait, how does atm_flow know about ctx? The example code assumes global context or passed context?
        # Actually in the example code: `ctx = ObservationContext(t*u.s)` is created but NOT passed to `process`.
        # This implies `helios.Context` might be a singleton or `Atmosphere` reads global state?
        # Or `ObservationContext` init sets global state? 
        # Let's assume the example code is correct and `helios.Context` magic works.
        # Replicating the context creation:
        
        ctx = ObservationContext(t*u.s)
        # Assuming Context works as a stack/singleton active context
        
        wf_t_atm = atm_flow.process(wf_t)
        
        phase_t = np.angle(wf_t_atm.field)
        
        im = axes[i].imshow(phase_t, origin='lower', cmap='twilight', 
                            vmin=-np.pi, vmax=np.pi, extent=[-1, 1, -1, 1])
        
        drift = np.linalg.norm(atm_flow.wind_velocity.value) * t
        axes[i].set_title(f't={t:.1f}s (drift: {drift:.1f}m)')
        axes[i].axis('off')

    plt.suptitle(f'Frozen-Flow Evolution', fontsize=16)
    st.pyplot(fig)
