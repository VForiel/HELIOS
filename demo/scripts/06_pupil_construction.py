"""
06_pupil_construction.py

Demonstrates how to construct telescope pupils using manual primitives and presets.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

import helios

def run_demo(save=False):
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
    
    if save:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated'))
        os.makedirs(output_dir, exist_ok=True)
        filename = "06_pupil_construction.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()
        
    print(f"Manual pupil shape: {arr_manual.shape}, fill factor: {arr_manual.mean():.3f}")

if __name__ == "__main__":
    run_demo()
