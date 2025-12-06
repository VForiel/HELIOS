import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from astropy import units as u
import helios


def test_collector_auto_magnify_true():
    wf = helios.Wavefront(wavelength=550*u.nm, size=2*u.m, npix=64, nsource=1)
    pupil = helios.Pupil(diameter=1.0*u.m)
    col = helios.Collector(pupil=pupil, position=(0,0), size=1.0*u.m)
    wf2 = col.process(wf, context=None, auto_magnify=True)
    assert np.isclose(wf2.width.to(u.m).value, 1.0)
    assert np.isclose(wf2.pixel_scale.to(u.m).value, 1.0/64)


def test_collector_auto_magnify_false_crop():
    wf = helios.Wavefront(wavelength=550*u.nm, size=2*u.m, npix=100, nsource=1)
    pupil = helios.Pupil(diameter=1.0*u.m)
    col = helios.Collector(pupil=pupil, position=(0,0), size=1.0*u.m)
    wf2 = col.process(wf, context=None, auto_magnify=False)
    assert np.isclose(wf2.width.to(u.m).value, 1.0)
    # pixel_scale should remain original (2m / 100 = 0.02m) but size changed by crop
    assert np.isclose(wf2.pixel_scale.to(u.m).value, 2.0/100)


def test_pupil_auto_magnify_none_warning():
    wf = helios.Wavefront(wavelength=550*u.nm, size=2*u.m, npix=64, nsource=1)
    p = helios.Pupil(diameter=1.0*u.m)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wf2 = p.process(wf, auto_magnify=None)
        assert len(w) >= 1
        assert np.isclose(wf2.width.to(u.m).value, 1.0)


def test_coronagraph_auto_magnify():
    wf = helios.Wavefront(wavelength=550*u.nm, size=2*u.m, npix=64, nsource=1)
    c = helios.Coronagraph(phase_mask='4quadrants', diameter=1.0*u.m)
    wf2 = c.process(wf, context=None, auto_magnify=True)
    assert np.isclose(wf2.width.to(u.m).value, 1.0)
    assert np.isclose(wf2.pixel_scale.to(u.m).value, 1.0/64)
