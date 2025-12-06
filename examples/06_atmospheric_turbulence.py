"""
06_atmospheric_turbulence.py

Demonstrates atmospheric turbulence effects:
1. Frozen-flow temporal evolution (6 snapshots) using a JWST pupil.
2. Animation of turbulence drift over the VLTI array.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from IPython.display import HTML

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios

def run_demo():
    # 1. Frozen-Flow Evolution (JWST Pupil)
    print("Simulating frozen-flow evolution (JWST Pupil)...")
    
    class ObservationContext(helios.Context):
        def __init__(self, time):
            super().__init__()
            self.time = time

    atm_flow = helios.Atmosphere(rms=100*u.nm, wind_speed=10*u.m/u.s, wind_direction=45, seed=123)
    wavelength = 550e-9 * u.m
    N = 512
    pupil_jwst = helios.Pupil.like('JWST')
    p_amp = pupil_jwst.get_array(npix=N, soft=True)
    
    times = [0, 0.5, 1.0, 1.5, 2.0, 2.5]  # seconds

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(times):
        wf_t = helios.Wavefront(wavelength=wavelength, npix=N)
        wf_t.field = p_amp.astype(np.complex128)
        
        ctx = ObservationContext(t*u.s)
        wf_t_atm = atm_flow.process(wf_t, ctx)
        
        phase_t = np.angle(wf_t_atm.field)
        
        im = axes[i].imshow(phase_t, origin='lower', cmap='twilight', 
                            vmin=-np.pi, vmax=np.pi, extent=[-1, 1, -1, 1])
        axes[i].set_title(f't={t:.1f}s (wind drift: {np.linalg.norm(atm_flow.wind_velocity)*t:.1f}m)')
        axes[i].set_xlabel('Normalized pupil')
        axes[i].set_ylabel('Normalized pupil')

    plt.suptitle(f'Frozen-Flow Atmospheric Evolution (λ={wavelength.to(u.nm).value:.0f}nm)', fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Phase (radians)', fontsize=11)
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated/examples'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    # 2. Animation (VLTI Array)
    print("Generating animation (VLTI Array)...")
    
    # Use VLTI preset (4 UTs)
    vlti = helios.TelescopeArray.vlti(uts=True)

    atm_anim = helios.Atmosphere(rms=150*u.nm, wind_speed=12*u.m/u.s, wind_direction=30, seed=456)
    anim_vlti = atm_anim.plot_animation(
        collectors=vlti, 
        duration=5*u.s,
        wavelength=550e-9*u.m,
        npix=256,
        fps=30,
        figsize=(8, 8)
    )
    
    try:
        anim_vlti.save('animation_vlti.mp4', writer='ffmpeg')
        print("Saved animation_vlti.mp4")
    except Exception as e:
        print(f"Could not save MP4 (ffmpeg might be missing): {e}")
        print("Animation object created successfully.")

if __name__ == "__main__":
    run_demo()
