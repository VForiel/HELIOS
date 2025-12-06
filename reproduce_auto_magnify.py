import sys
sys.path.insert(0, 'src')
import helios
from astropy import units as u
import numpy as np
import warnings

def test_auto_magnify():
    print("Testing Wavefront refactor and auto_magnify...")
    
    # 1. Test Wavefront creation
    print("\n1. Testing Wavefront creation...")
    wf = helios.Wavefront(wavelength=550*u.nm, size=10*u.m, npix=100, nsource=1)
    assert wf.size == 10*u.m
    assert wf.npix == 100
    assert wf.pixel_scale == 0.1*u.m
    print("   Wavefront created successfully.")
    
    # 2. Test Collector with auto_magnify=True
    print("\n2. Testing Collector with auto_magnify=True...")
    pupil = helios.Pupil(diameter=5*u.m)
    collector = helios.Collector(pupil=pupil, size=5*u.m, position=(0,0))
    
    wf_large = helios.Wavefront(wavelength=550*u.nm, size=10*u.m, npix=100, nsource=1)
    # Should resize wf to 5m, keeping npix=100 -> pixel_scale = 0.05m
    wf_processed = collector.process(wf_large, None, auto_magnify=True)
    
    assert wf_processed.size == 5*u.m
    assert wf_processed.npix == 100
    assert wf_processed.pixel_scale == 0.05*u.m
    print("   auto_magnify=True worked: Size changed from 10m to 5m.")
    
    # 3. Test Collector with auto_magnify=False (Crop)
    print("\n3. Testing Collector with auto_magnify=False (Crop)...")
    wf_large = helios.Wavefront(wavelength=550*u.nm, size=10*u.m, npix=100, nsource=1)
    # Should crop wf to 5m. 
    # Original pixel scale = 0.1m. 5m corresponds to 50 pixels.
    wf_processed = collector.process(wf_large, None, auto_magnify=False)
    
    assert wf_processed.size == 5*u.m
    assert wf_processed.npix == 50
    assert wf_processed.pixel_scale == 0.1*u.m
    print("   auto_magnify=False worked: Cropped to 50 pixels.")
    
    # 4. Test Collector with auto_magnify=None (Warning + Resize)
    print("\n4. Testing Collector with auto_magnify=None (Warning)...")
    wf_large = helios.Wavefront(wavelength=550*u.nm, size=10*u.m, npix=100, nsource=1)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wf_processed = collector.process(wf_large, None, auto_magnify=None)
        assert len(w) > 0
        assert "does not match Collector size" in str(w[-1].message)
        print("   Warning caught successfully.")
        
    assert wf_processed.size == 5*u.m
    assert wf_processed.npix == 100 # Resized metadata, not cropped
    print("   Default behavior worked: Resized metadata.")

if __name__ == "__main__":
    test_auto_magnify()    # 5. Test Pupil with auto_magnify=False (Crop)
    print('
5. Testing Pupil with auto_magnify=False (Crop)...')
    pupil = helios.Pupil(diameter=5*u.m)
    wf_large = helios.Wavefront(wavelength=550*u.nm, size=10*u.m, npix=100, nsource=1)
    wf_processed = pupil.process(wf_large, auto_magnify=False)
    assert wf_processed.size == 5*u.m
    assert wf_processed.npix == 50
    print('   Pupil auto_magnify=False worked.')
    if __name__ == '__main__': test_auto_magnify()
