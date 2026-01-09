
"""
01_propagation_comparisons.py

This script demonstrates and compares different optical propagation methods available in HELIOS.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import helios
from helios import Wavefront

def run_demo(save=False):
    print("=== HELIOS Propagation Comparison Demo ===")
    
    # --- 1. Parameters ---
    wavelength = 633 * u.nm
    size = 2.0 * u.mm         # Pupil diameter
    npix_in = 256
    
    focal_length = 50 * u.mm  # Lens focal length
    
    scenarios = [
        {"name": "Focal Plane (z=f)", "distance": focal_length, "use_lens": True},
        {"name": "Near Field (z=f/2)", "distance": focal_length / 2, "use_lens": True},
        {"name": "Far Field (z=10f)", "distance": focal_length * 10, "use_lens": True},
        {"name": "Free Space (No Lens)", "distance": 10*size, "use_lens": False},
    ]
    
    methods_of_interest = [
        'Fraunhofer', 'Fresnel', 'ASM', 'SCASM',
        'Poppy', 'HCIPy', 'LightPipes',
        'dLux_ASM', 'dLux_MFT', 'dLux_FFT'
    ]

    # --- 2. Input Wavefront Setup ---
    print(f"\nInitializing Input Wavefront:")
    print(f"  Wavelength: {wavelength}")
    print(f"  Size: {size}")
    print(f"  Resolution: {npix_in}x{npix_in}")
    
    wf_in = Wavefront(wavelength=wavelength, size=size, npix=npix_in)
    
    # Circular Aperture
    y, x = wf_in.coordinates()
    r = np.sqrt(x**2 + y**2)
    mask = r <= (size / 2)
    wf_in[:] = mask.astype(complex)
    
    a = size / 2
    
    # --- 3. Loop over Scenarios ---
    for scenario in scenarios:
        z = scenario["distance"]
        use_lens = scenario["use_lens"]
        name = scenario["name"]
        
        print(f"\n--- Scenario: {name} ---")
        print(f"  Distance z: {z}")
        N_F = (a**2 / (z * wavelength)).decompose()
        print(f"  Fresnel Number: {N_F:.2f}")
        
        # Prepare figure
        n_methods = len(methods_of_interest) + 1 # +1 for Input
        cols = 4
        rows = (n_methods + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.atleast_1d(axes).flatten()
        
        # 1. Plot Input Wavefront first
        ax_in = axes[0]
        img_in = wf_in.intensity
        extent_in, xl_in, yl_in = helios.core.wavefront.get_smart_extent(wf_in.shape, wf_in.pixel_scale)
        ax_in.imshow(img_in, extent=extent_in, cmap='gray', origin='lower')
        ax_in.set_title(f"Input")
        ax_in.set_xlabel(xl_in)
        ax_in.set_ylabel(yl_in)

        for i, method in enumerate(methods_of_interest):
            ax = axes[i+1] # Shift by one
            try:
                # Reset wavefront
                wf = wf_in.copy()
                f_arg = focal_length if use_lens else None
                output_npix = 256
                
                wf_out = wf.propagate(
                    distance=z,
                    focal_length=f_arg,
                    output_npix=output_npix,
                    regime=method
                )
                
                # Visualization
                img = wf_out.intensity
                img_log = np.log10(img + 1e-12)
                extent, xl, yl = helios.core.wavefront.get_smart_extent(wf_out.shape, wf_out.pixel_scale)
                
                im = ax.imshow(img_log, extent=extent, cmap='inferno', origin='lower')
                ax.set_title(f"{method}")
                ax.set_xlabel(xl)
                
                # Compute Energy conservation
                ratio = wf_out.integrated_intensity / wf_in.integrated_intensity
                ax.text(0.05, 0.95, f"E_ratio: {ratio:.2f}", transform=ax.transAxes, color='white', fontsize=8, va='top')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', color='red', transform=ax.transAxes)
                ax.set_title(f"{method} (Failed)")
                # print(f"  {method} failed: {e}") # Reduce logging noise
        
        # Hide unused subplots
        for j in range(n_methods, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(f"Scenario: {name} (z={z}, N_F={N_F:.1f})")
        plt.tight_layout()
        
        if save:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated'))
            os.makedirs(output_dir, exist_ok=True)
            
            # Sanitize name
            safe_name = name.replace(" ", "_").lower()
            for char in ['/', '\\', ':', '(', ')', '=']:
                safe_name = safe_name.replace(char, '')
                
            filename = f"01_propagation_comparisons_{safe_name}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close() # Important to close memory
        else:
            plt.show()

if __name__ == "__main__":
    run_demo()
