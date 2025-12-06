"""Test suite for Wavefront.adapt() method."""
import sys
sys.path.insert(0, '../src')

import numpy as np
from astropy import units as u
import helios


def test_wavefront_pixel_scale_always_defined():
    """Test that pixel_scale is always defined in Wavefront.__init__."""
    wf = helios.Wavefront(size=1*u.m, npix=256)
    assert hasattr(wf, 'pixel_scale'), "pixel_scale should always be defined"
    assert isinstance(wf.pixel_scale, u.Quantity), "pixel_scale should be a Quantity"
    assert wf.pixel_scale.unit.is_equivalent(u.m), "pixel_scale should be in meters"
    
    # Test with value parameter
    field = np.ones((1, 128, 128), dtype=np.complex128)
    wf2 = helios.Wavefront(size=2*u.m, value=field)
    assert hasattr(wf2, 'pixel_scale'), "pixel_scale should be defined even with value parameter"
    assert wf2.pixel_scale == (2*u.m / 128), "pixel_scale should be correctly computed"


def test_adapt_crop_mode():
    """Test adapt() in crop mode (magnify=False)."""
    wf = helios.Wavefront(size=2*u.m, npix=512)
    wf_adapted = wf.adapt(size=1*u.m, magnify=False)
    
    assert wf_adapted.width == 1*u.m, "Size should be 1m after crop"
    assert wf_adapted.npix == 256, "Should be cropped to half the pixels"
    assert np.isclose(wf_adapted.pixel_scale.to(u.m).value, 
                      wf.pixel_scale.to(u.m).value, rtol=1e-5), \
        "pixel_scale should remain the same after crop"


def test_adapt_magnify_mode():
    """Test adapt() in magnify mode (magnify=True)."""
    wf = helios.Wavefront(size=2*u.m, npix=512)
    wf_adapted = wf.adapt(size=1*u.m, magnify=True)
    
    assert wf_adapted.width == 1*u.m, "Size should be 1m after magnify"
    assert wf_adapted.npix == 512, "npix should remain unchanged in magnify mode"
    assert wf_adapted.pixel_scale == 1*u.m / 512, "pixel_scale should be updated"


def test_adapt_resample():
    """Test adapt() with resampling (npix parameter)."""
    wf = helios.Wavefront(size=2*u.m, npix=512)
    wf[:] = 1.0
    
    wf_resampled = wf.adapt(size=wf.width, npix=256)
    
    assert wf_resampled.width == 2*u.m, "Size should remain 2m"
    assert wf_resampled.npix == 256, "Should be resampled to 256 pixels"
    assert wf_resampled.shape == (1, 256, 256), "Field shape should be (1, 256, 256)"
    assert wf_resampled.pixel_scale == 2*u.m / 256, "pixel_scale should be updated for new resolution"


def test_adapt_upscale():
    """Test adapt() with upscaling."""
    wf = helios.Wavefront(size=1*u.m, npix=128)
    wf[:] = 1.0
    
    wf_upscaled = wf.adapt(size=wf.width, npix=512)
    
    assert wf_upscaled.npix == 512, "Should be upscaled to 512 pixels"
    assert wf_upscaled.shape == (1, 512, 512), "Field shape should be (1, 512, 512)"
    # Check that upscaling preserves total flux (approximately)
    original_flux = np.sum(np.abs(wf)**2)
    upscaled_flux = np.sum(np.abs(wf_upscaled)**2)
    # After zoom, flux is scaled by zoom_factor^2
    assert np.isclose((upscaled_flux / original_flux).value, (512/128)**2, rtol=0.01)


def test_pupil_process_with_adapt():
    """Test Pupil.process() uses adapt() correctly."""
    wf = helios.Wavefront(size=2*u.m, npix=512)
    wf[:] = 1.0
    
    pupil = helios.Pupil(diameter=1*u.m)
    pupil.add_disk(radius=0.4*u.m)  # Disk smaller than diameter to create zeros at edges
    
    wf_processed = pupil.process(wf, auto_magnify=True)
    
    # Verify the wavefront was adapted to pupil size
    assert wf_processed.width == 1*u.m, "Wavefront should be adapted to pupil diameter"
    assert wf_processed.npix == 512, "npix should be preserved in magnify mode"
    assert hasattr(wf_processed, 'pixel_scale'), "pixel_scale should exist"
    
    # Verify pupil mask was applied (check corners which should be zero)
    # In a circular aperture, corners are always zero
    corner_value = np.abs(wf_processed[0, 0, 0])
    assert corner_value < 0.5, "Pupil mask should create zeros in corners"


def test_adapt_preserves_original():
    """Test that adapt() doesn't modify the original wavefront."""
    wf = helios.Wavefront(size=2*u.m, npix=512)
    original_size = wf.width
    original_npix = wf.npix
    original_field_id = id(wf)
    
    wf_adapted = wf.adapt(size=1*u.m, magnify=True, npix=256)
    
    # Original should be unchanged
    assert wf.width == original_size, "Original size should be unchanged"
    assert wf.npix == original_npix, "Original npix should be unchanged"
    assert id(wf) == original_field_id, "Original field should not be replaced"
    
    # Adapted should be different
    assert wf_adapted.width != wf.width or wf_adapted.npix != wf.npix


if __name__ == "__main__":
    print("Running Wavefront.adapt() test suite...")
    test_wavefront_pixel_scale_always_defined()
    print("✓ pixel_scale always defined")
    
    test_adapt_crop_mode()
    print("✓ adapt() crop mode")
    
    test_adapt_magnify_mode()
    print("✓ adapt() magnify mode")
    
    test_adapt_resample()
    print("✓ adapt() resample")
    
    test_adapt_upscale()
    print("✓ adapt() upscale")
    
    test_pupil_process_with_adapt()
    print("✓ Pupil.process() integration")
    
    test_adapt_preserves_original()
    print("✓ adapt() preserves original")
    
    print("\n✓ All tests passed!")
