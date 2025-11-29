import sys
import os
import numpy as np
from astropy import units as u

# ensure local `src` is first on path so tests import the workspace code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from helios.components.pupil import Pupil
from helios.components.collector import TelescopeArray
from helios.core.simulation import Wavefront


def test_telescope_array_applies_pupil_mask():
    # small pupil and wavefront for fast test
    p = Pupil(1 * u.m)
    # filled disk that should zero-out outer region
    p.add_disk(radius=0.5)

    array = TelescopeArray()
    array.add_collector(pupil=p, position=(0, 0), size=1 * u.m)

    wf = Wavefront(wavelength=600 * u.nm, size=128)
    # start with uniform amplitude ones
    wf.field = np.ones_like(wf.field, dtype=complex)

    wf2 = array.process(wf, None)

    # get expected mask at same sampling
    mask = p.get_array(npix=128, soft=True)

    expected_zero = (mask == 0.0)
    actual_zero = (np.abs(wf2.field) < 1e-12)

    # number of zeroed pixels should match
    assert actual_zero.sum() == expected_zero.sum()
