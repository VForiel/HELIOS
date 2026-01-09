"""
03_pupil_construction.py

Demonstrates how to construct telescope pupils using manual primitives and presets.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import helios

def run_demo():
    # 1. Manual pupil construction
    print("Constructing manual pupil...")
    p_manual = helios.Pupil(8*u.m)
    p_manual.add_disk(radius=4.0 * u.m)
    p_manual.add_central_obscuration(diameter=1.1 * u.m)
    p_manual.add_spiders(arms=4, width=0.05 * u.m)

    # 2. JWST Preset
    print("Loading JWST preset...")
    p_jwst = helios.Pupil.like('JWST')

    # Visualize pupils
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    arr_manual = p_manual.get_array(npix=512)
    axes[0].imshow(arr_manual, origin='lower', cmap='gray')
    axes[0].set_title('Manual Pupil (8m telescope)')
    axes[0].axis('off')

    arr_jwst = p_jwst.get_array(npix=512)
    axes[1].imshow(arr_jwst, origin='lower', cmap='gray')
    axes[1].set_title('JWST Pupil (preset)')
    axes[1].axis('off')

    plt.tight_layout()
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
        
    print(f"Manual pupil shape: {arr_manual.shape}, fill factor: {arr_manual.mean():.3f}")

if __name__ == "__main__":
    run_demo()
