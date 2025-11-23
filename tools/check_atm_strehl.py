import sys
import numpy as np
from astropy import units as u
# ensure local src is prefered when running from repo root
sys.path.insert(0, 'src')
from helios.components import Pupil, Atmosphere, AdaptiveOptics
from helios.core.simulation import Wavefront

wavelength = 550e-9 * u.m
N = 256
pupil = Pupil.like('JWST')
p_amp = pupil.get_array(npix=N, soft=True)

# ideal
wf_ideal = Wavefront(wavelength=wavelength, size=N)
wf_ideal.field = p_amp.astype(np.complex128)
field_ideal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf_ideal.field)))
psf_ideal = np.abs(field_ideal)**2
peak_ideal = float(psf_ideal.max())

# atmosphere
wf = Wavefront(wavelength=wavelength, size=N)
wf.field = p_amp.astype(np.complex128)
atm = Atmosphere(rms=0.5, seed=1)
wf_atm = atm.process(wf, None)
field_atm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf_atm.field)))
psf_atm = np.abs(field_atm)**2
peak_atm = float(psf_atm.max())

# AO
ao = AdaptiveOptics(coeffs={2:0.3})
wf_corr = Wavefront(wavelength=wavelength, size=N)
wf_corr.field = wf_atm.field.copy()
wf_after = ao.process(wf_corr, None)
field_ao = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf_after.field)))
psf_ao = np.abs(field_ao)**2
peak_ao = float(psf_ao.max())

strehl_atm = peak_atm / peak_ideal if peak_ideal>0 else float('nan')
strehl_ao = peak_ao / peak_ideal if peak_ideal>0 else float('nan')

print(f"peak_ideal={peak_ideal:.6e}")
print(f"peak_atm={peak_atm:.6e}")
print(f"peak_ao={peak_ao:.6e}")
print(f"strehl_atm={strehl_atm:.6f}")
print(f"strehl_ao={strehl_ao:.6f}")
