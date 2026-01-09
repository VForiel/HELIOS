"""
10_chromatic_psf.py

Demonstrates chromatic PSF degradation due to atmospheric turbulence.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

import helios

def run_demo(save=False):
    N = 512
    pupil = helios.Pupil.like('JWST')
    p_amp = pupil.get_array(npix=N, soft=True)

    test_cases = [
        (400, 50, "λ=400nm (blue), OPD=50nm"),
        (550, 50, "λ=550nm (visible), OPD=50nm"),
        (400, 100, "λ=400nm (blue), OPD=100nm"),
        (550, 100, "λ=550nm (visible), OPD=100nm"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (wl_nm, opd_nm, title) in enumerate(test_cases):
        wavelength = wl_nm * 1e-9 * u.m
        
        # Ideal PSF
        wf_ideal = helios.Wavefront(wavelength=wavelength, npix=N)
        wf_ideal.field = p_amp.astype(np.complex128)
        field_ideal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf_ideal.field)))
        psf_ideal = np.abs(field_ideal) ** 2
        peak_ideal = psf_ideal.max()
        
        # Degraded PSF
        wf = helios.Wavefront(wavelength=wavelength, npix=N)
        wf.field = p_amp.astype(np.complex128)
        
        atm = helios.Atmosphere(rms=opd_nm * u.nm, seed=42)
        wf_atm = atm.process(wf)
        
        field_final = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf_atm.field)))
        psf_final = np.abs(field_final) ** 2
        
        # Normalize
        psf_norm_final = psf_final / peak_ideal
        psf_norm_ideal = psf_ideal / peak_ideal
        
        strehl = psf_final.max() / peak_ideal
        phase_rms_rad = 2 * np.pi * (opd_nm * 1e-9) / (wl_nm * 1e-9)
        
        # Plot PSF
        axes[i].imshow(np.log10(psf_norm_final + 1e-10), origin='lower', cmap='inferno')
        axes[i].set_title(f"{title}\nStrehl={strehl:.3f}, φ_rms={phase_rms_rad:.2f}rad")
        axes[i].axis('off')
        
        # Plot Difference
        diff = psf_norm_ideal - psf_norm_final
        axes[i+4].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[i+4].set_title("Difference (ideal - degraded)")
        axes[i+4].axis('off')

    plt.tight_layout()
    
    if save:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated'))
        os.makedirs(output_dir, exist_ok=True)
        filename = "10_chromatic_psf.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    run_demo()
