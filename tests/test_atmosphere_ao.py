import sys
import os
import numpy as np
from astropy import units as u

# ensure local `src` is first on path so tests import the workspace code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from helios.core.simulation import Wavefront
from helios.components import Atmosphere, AdaptiveOptics


def test_atmosphere_changes_phase_only():
    wf = Wavefront(wavelength=600 * u.nm, npix=128)
    orig_amp = np.abs(wf).copy()
    atm = Atmosphere(rms=0.5, seed=42)
    wf2 = atm.process(wf)
    # amplitude should remain roughly the same (pure phase)
    assert wf2.shape == (1, 128, 128)
    assert np.allclose(np.abs(wf2), orig_amp)


def test_ao_zernike_coefficients_apply():
    wf = Wavefront(wavelength=600 * u.nm, npix=128)
    wf[:] *= np.exp(1j * 0.3)  # add global phase
    ao = AdaptiveOptics(coeffs={(0, 0): 0.1})
    wf_before = wf.copy()
    wf2 = ao.process(wf)
    # With non-zero coefficient, field changes
    assert not np.allclose(wf_before, wf2)
    # If coefficients are zero, field stays the same
    wf3 = Wavefront(wavelength=600 * u.nm, npix=128)
    ao_zero = AdaptiveOptics(coeffs={(0, 0): 0.0})
    wf3b = ao_zero.process(wf3)
    assert np.allclose(wf3, wf3b)


def test_ao_noll_index_support():
    wf = Wavefront(wavelength=600 * u.nm, npix=64)
    # set a small Zernike via Noll index 2 (tilt)
    ao = AdaptiveOptics(coeffs={2: 0.05})
    wf_before = wf.copy()
    wf2 = ao.process(wf)
    assert not np.allclose(wf_before, wf2)
