import numpy as np
from astropy import units as u

from helios.core.simulation import Wavefront
from helios.components.optics import Atmosphere, AdaptiveOptics


def test_atmosphere_changes_phase_only():
    wf = Wavefront(wavelength=600 * u.nm, size=128)
    orig_amp = np.abs(wf.field).copy()
    atm = Atmosphere(rms=0.5, seed=42)
    wf2 = atm.process(wf, None)
    # amplitude should remain roughly the same (pure phase)
    assert wf2.field.shape == (128, 128)
    assert np.allclose(np.abs(wf2.field), orig_amp)


def test_ao_zernike_coefficients_apply():
    wf = Wavefront(wavelength=600 * u.nm, size=128)
    wf.field *= np.exp(1j * 0.3)  # add global phase
    ao = AdaptiveOptics(coeffs={(0, 0): 0.1})
    wf_before = wf.field.copy()
    wf2 = ao.process(wf, None)
    # With non-zero coefficient, field changes
    assert not np.allclose(wf_before, wf2.field)
    # If coefficients are zero, field stays the same
    wf3 = Wavefront(wavelength=600 * u.nm, size=128)
    ao_zero = AdaptiveOptics(coeffs={(0, 0): 0.0})
    wf3b = ao_zero.process(wf3, None)
    assert np.allclose(wf3.field, wf3b.field)


def test_ao_noll_index_support():
    wf = Wavefront(wavelength=600 * u.nm, size=64)
    # set a small Zernike via Noll index 2 (tilt)
    ao = AdaptiveOptics(coeffs={2: 0.05})
    wf_before = wf.field.copy()
    wf2 = ao.process(wf, None)
    assert not np.allclose(wf_before, wf2.field)
