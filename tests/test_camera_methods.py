"""
Test Camera class with new acquisition and reduction methods.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from astropy import units as u
import helios


def test_camera_instantiation():
    """Test Camera instantiation with various parameters."""
    # Default parameters
    cam = helios.Camera()
    assert cam.pixels == (1024, 1024)
    assert cam.dark_current == 0.01  # e-/s (converted to float)
    assert cam.read_noise == 3.0  # e-
    assert cam.integration_time_value == 1.0  # s
    assert cam.quantum_efficiency == 0.9
    assert cam.gain == 1.0
    
    # Custom parameters
    cam2 = helios.Camera(
        pixels=(512, 512),
        dark_current=0.05 * u.electron / u.s,
        read_noise=5 * u.electron,
        integration_time=10 * u.s,
        quantum_efficiency=0.85,
        gain=2.0
    )
    assert cam2.pixels == (512, 512)
    assert cam2.dark_current == 0.05
    assert cam2.read_noise == 5.0
    assert cam2.integration_time_value == 10.0
    assert cam2.quantum_efficiency == 0.85
    assert cam2.gain == 2.0
    
    print("✓ Camera instantiation tests passed")


def test_dark_frame():
    """Test dark frame generation."""
    cam = helios.Camera(
        pixels=(256, 256),
        dark_current=0.1 * u.electron / u.s,
        read_noise=2 * u.electron,
        integration_time=100 * u.s
    )
    
    dark = cam.get_dark()
    
    # Check shape
    assert dark.shape == (256, 256), f"Expected (256, 256), got {dark.shape}"
    
    # Check that dark has reasonable values
    # Expected dark electrons: 0.1 e-/s × 100s = 10 e-/pixel
    # With shot noise and read noise, mean should be close to 10 e-
    assert 5 < dark.mean() < 15, f"Dark mean {dark.mean():.1f} out of expected range"
    
    # Standard deviation should include shot noise (~√10) and read noise (~2)
    # Total σ ≈ √(10 + 4) ≈ 3.7
    assert 2 < dark.std() < 6, f"Dark std {dark.std():.1f} out of expected range"
    
    print(f"✓ Dark frame: mean={dark.mean():.2f} e-, std={dark.std():.2f} e-")


def test_raw_image_no_signal():
    """Test raw image with no wavefront (should be equivalent to dark)."""
    cam = helios.Camera(
        pixels=(128, 128),
        dark_current=0.05 * u.electron / u.s,
        integration_time=50 * u.s
    )
    
    raw = cam.get_raw_image(wavefront=None)
    dark = cam.get_dark()
    
    # Both should have same shape
    assert raw.shape == dark.shape == (128, 128)
    
    # Statistics should be similar (both are dark frames with noise)
    # Not exactly equal because of random noise
    assert abs(raw.mean() - dark.mean()) < 2, "Raw and dark means should be close"
    assert abs(raw.std() - dark.std()) < 1, "Raw and dark stds should be close"
    
    print("✓ Raw image without signal matches dark frame statistics")


def test_raw_image_with_signal():
    """Test raw image acquisition with wavefront."""
    cam = helios.Camera(
        pixels=(128, 128),
        dark_current=0.01 * u.electron / u.s,
        integration_time=10 * u.s,
        quantum_efficiency=0.9
    )
    
    # Create a wavefront with uniform intensity
    wf = helios.Wavefront(wavelength=550e-9 * u.m, size=128)
    wf.field = np.ones((128, 128), dtype=np.complex128) * 10  # Amplitude = 10
    
    raw = cam.get_raw_image(wf)
    
    # Check shape
    assert raw.shape == (128, 128)
    
    # Expected signal: |field|² × QE × t = 100 × 0.9 × 10 = 900 e-/pixel
    # Expected dark: 0.01 × 10 = 0.1 e-/pixel (negligible)
    # Total ≈ 900 e-/pixel
    assert 800 < raw.mean() < 1000, f"Raw mean {raw.mean():.1f} out of expected range"
    
    # Shot noise should dominate: σ ≈ √900 ≈ 30
    assert 20 < raw.std() < 40, f"Raw std {raw.std():.1f} out of expected range"
    
    print(f"✓ Raw image with signal: mean={raw.mean():.1f} e-, std={raw.std():.1f} e-")


def test_image_reduction():
    """Test automatic dark subtraction."""
    cam = helios.Camera(
        pixels=(128, 128),
        dark_current=1.0 * u.electron / u.s,  # High dark current to make it visible
        read_noise=5 * u.electron,
        integration_time=10 * u.s,
        quantum_efficiency=0.9
    )
    
    # Create wavefront with signal
    wf = helios.Wavefront(wavelength=550e-9 * u.m, size=128)
    wf.field = np.ones((128, 128), dtype=np.complex128) * 10  # |field|² = 100
    
    # Get reduced image (with dark subtraction)
    reduced = cam.get_image(wf, subtract_dark=True)
    
    # Get raw image (without dark subtraction)
    raw = cam.get_image(wf, subtract_dark=False)
    
    # Raw should have more signal than reduced (contains dark)
    # Expected dark: 1.0 e-/s × 10s = 10 e-/pixel
    # So raw.mean() ≈ reduced.mean() + 10
    assert raw.mean() > reduced.mean(), "Raw should have more electrons than reduced"
    
    # Expected signal after reduction: 100 × 0.9 × 10 = 900 e-/pixel
    assert 800 < reduced.mean() < 1000, f"Reduced mean {reduced.mean():.1f} out of expected range"
    
    print(f"✓ Image reduction: raw={raw.mean():.1f} e-, reduced={reduced.mean():.1f} e-")
    print(f"  Dark contribution: ~{raw.mean() - reduced.mean():.1f} e-")


def test_process_method():
    """Test that process() returns reduced image."""
    cam = helios.Camera(pixels=(64, 64))
    
    wf = helios.Wavefront(wavelength=550e-9 * u.m, size=64)
    wf.field = np.ones((64, 64), dtype=np.complex128)
    
    # process() should return reduced image by default
    result = cam.process(wf, None)
    
    assert result.shape == (64, 64)
    assert isinstance(result, np.ndarray)
    
    # Should be equivalent to get_image with subtract_dark=True
    reduced = cam.get_image(wf, None, subtract_dark=True)
    
    # Won't be exactly equal due to random noise, but should be statistically similar
    assert abs(result.mean() - reduced.mean()) < 5
    
    print("✓ process() method returns reduced image")


def test_noise_statistics():
    """Test that noise follows expected statistics."""
    cam = helios.Camera(
        pixels=(512, 512),
        dark_current=0 * u.electron / u.s,  # No dark current
        read_noise=10 * u.electron,
        integration_time=1 * u.s,
        quantum_efficiency=1.0  # Perfect QE
    )
    
    # Create wavefront with known signal
    wf = helios.Wavefront(wavelength=550e-9 * u.m, size=512)
    signal_level = 1000  # photons
    wf.field = np.ones((512, 512), dtype=np.complex128) * np.sqrt(signal_level)
    
    # Get raw image
    raw = cam.get_raw_image(wf)
    
    # Expected: signal_level electrons per pixel
    # Shot noise: √signal_level ≈ 31.6
    # Read noise: 10
    # Total noise: √(31.6² + 10²) ≈ 33.2
    
    expected_mean = signal_level
    expected_std = np.sqrt(signal_level + 10**2)
    
    assert abs(raw.mean() - expected_mean) / expected_mean < 0.05, \
        f"Mean {raw.mean():.1f} deviates from expected {expected_mean}"
    
    assert abs(raw.std() - expected_std) / expected_std < 0.1, \
        f"Std {raw.std():.1f} deviates from expected {expected_std:.1f}"
    
    print(f"✓ Noise statistics: expected σ={expected_std:.1f}, measured σ={raw.std():.1f}")


if __name__ == "__main__":
    test_camera_instantiation()
    test_dark_frame()
    test_raw_image_no_signal()
    test_raw_image_with_signal()
    test_image_reduction()
    test_process_method()
    test_noise_statistics()
    print("\n✅ All Camera method tests passed!")
