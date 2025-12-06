
import sys
import os
import numpy as np
import astropy.units as u

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import helios

def debug_visibility():
    print("--- Debugging Visibility ---")

    # 1. Setup Scene
    star = helios.Star(magnitude=5, temperature=5778*u.K)
    scene = helios.Scene(distance=10*u.pc)
    scene.add(star)
    
    # 2. Setup Telescope
    pupil = helios.Pupil(diameter=1*u.m)
    # Simple circular pupil for clarity
    
    collector = helios.Collector(pupil=pupil, position=[0, 0]*u.m)
    
    # 3. Get Input Wavefront
    ctx = helios.Context()
    ctx.add_layer(scene)
    # We pass collectors explicitly to mimic what happens inside Context when TelescopeArray is used
    # But here we just want one wavefront to analyze
    
    wavelength = 1e-6 * u.m
    size = 1024
    
    print(f"Generating input wavefront for wavelength: {wavelength}, size: {size}")
    # We use the internal logic of get_input_wavefront by passing collectors
    wf_array = ctx.get_input_wavefront(wavelength=wavelength, size=size, collectors=[collector])
    wf = wf_array.wavefronts[0]
    
    print(f"Pupil Plane Pixel Scale: {wf.pixel_scale}")
    print(f"Pupil Plane Total Energy: {np.sum(np.abs(wf.field)**2)}")
    
    # 4. Propagate with Padding
    padding = 4
    print(f"Propagating with padding={padding}...")
    wf.propagate(distance=10*u.m, padding=padding)
    
    print(f"Focal Plane Pixel Scale: {wf.pixel_scale}")
    
    # 5. Analyze Focal Plane
    intensity = np.abs(wf.field[0])**2
    max_val = np.max(intensity)
    mean_val = np.mean(intensity)
    sum_val = np.sum(intensity)
    
    print(f"Max Intensity: {max_val}")
    print(f"Mean Intensity: {mean_val}")
    print(f"Total Energy: {sum_val}")
    print(f"Dynamic Range (Max/Mean): {max_val/mean_val}")
    
    # Check central peak
    cy, cx = intensity.shape[0]//2, intensity.shape[1]//2
    center_crop = intensity[cy-5:cy+6, cx-5:cx+6]
    print("Central 11x11 pixels:")
    print(center_crop)
    
    # Check if peak is at center
    py, px = np.unravel_index(np.argmax(intensity), intensity.shape)
    print(f"Peak Location: ({px}, {py})")
    print(f"Center Location: ({cx}, {cy})")
    
    # Count pixels above thresholds
    n_above_half = np.sum(intensity > max_val/2)
    n_above_100 = np.sum(intensity > max_val/100)
    n_above_10000 = np.sum(intensity > max_val/10000)
    
    print(f"Pixels > Max/2 (FWHM area): {n_above_half}")
    print(f"Pixels > Max/100: {n_above_100}")
    print(f"Pixels > Max/10000: {n_above_10000}")

    # Save cropped image
    import matplotlib.pyplot as plt
    
    cy, cx = intensity.shape[0]//2, intensity.shape[1]//2
    crop_size = 50
    crop = intensity[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(np.log10(crop + 1e-12), origin='lower', cmap='inferno')
    plt.title(f"Central {2*crop_size}x{2*crop_size} pixels (Log)")
    plt.colorbar()
    plt.savefig("debug_visibility_crop.png")
    print("Saved debug_visibility_crop.png")

if __name__ == "__main__":
    debug_visibility()
