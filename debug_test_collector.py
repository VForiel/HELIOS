import sys
sys.path.insert(0, 'src')
import helios
from helios import Pupil, TelescopeArray, Wavefront
from astropy import units as u
import numpy as np

p = Pupil(1 * u.m)
p.add_disk(radius=0.5)

array = TelescopeArray()
array.add_collector(pupil=p, position=(0, 0), size=1 * u.m)

wf = Wavefront(wavelength=600 * u.nm, size=128)
print(f"Initial wavefront shape: {wf.field.shape}")
print(f"Initial wavefront size: {wf.size}")
print(f"Initial wavefront npix: {wf.npix}")

wf.field = np.ones_like(wf.field, dtype=complex)

wf2 = array.process(wf, None)
print(f"\nAfter TelescopeArray.process:")
print(f"Output type: {type(wf2)}")
print(f"Output field shape: {wf2.field.shape}")
print(f"Output size: {wf2.size}")
print(f"Output npix: {wf2.npix}")

mask = p.get_array(npix=128, soft=True)
print(f"\nMask shape: {mask.shape}")
print(f"Mask zeros: {(mask == 0).sum()} / {mask.size}")

expected_zero = (mask == 0.0)
actual_zero = (np.abs(wf2.field) < 1e-12)
print(f"\nExpected zeros: {expected_zero.sum()}")
print(f"Actual zeros: {actual_zero.sum()}")
