
import matplotlib.pyplot as plt
from  helios.components.pupil import Pupil
import numpy as np

def verify_padding():
    p = Pupil.vlt()
    
    # 1. Check array shape and padding visually
    npix = 100
    padding = 20
    arr = p.get_array(npix=npix, padding=padding)
    
    print(f"Array shape: {arr.shape} (Expected {npix}x{npix})")
    
    # Check that edges are zero (padding area)
    # With 20px padding on each side, the pupil (8.2m) should be in the center 60x60 pixels.
    # So pixels 0-19 and 80-99 should be 0.
    left_edge_max = np.max(arr[:, :19])
    right_edge_max = np.max(arr[:, -19:])
    
    print(f"Max value in left padding (0-19): {left_edge_max}")
    print(f"Max value in right padding (80-99): {right_edge_max}")
    
    if left_edge_max == 0 and right_edge_max == 0:
        print("SUCCESS: Padding appears empty.")
    else:
        print("FAILURE: Padding contains signal!")

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    p.plot(npix=npix, padding=padding, ax=plt.gca())
    plt.title(f"Pupil with padding={padding}")
    
    plt.subplot(122)
    # Check PSF axes
    # If padding works, the PSF scale (lambda/D) units must be preserved?
    # Actually, padding in pupil = oversampling in PSF ?? 
    # No, padding in pupil (zero padding) -> finer sampling in PSF (interpolation).
    # Wait, FFT of padded array -> PSF has better resolution in frequency domain?
    # No:
    # Larger spatial domain (Pupil plane size L) -> Finer resolution in Frequency domain (df = 1/L).
    # Here size_m is larger because of padding.
    # So PSF pixels will have smaller angular scale.
    # But plot_diffraction_pattern plots in units of lambda/D.
    # This scaling normalization relies on correct 'dx'.
    # If we messed up 'dx', the PSF size in lambda/D units will look wrong (e.g. airy ring not at 1.22).
    p.plot_diffraction_pattern(npix=256, padding=10, ax=plt.gca())
    plt.title("PSF (Check Airy rings)")
    
    plt.savefig("verify_padding_output.png")
    print("Saved plot verification to verify_padding_output.png")

if __name__ == "__main__":
    verify_padding()
