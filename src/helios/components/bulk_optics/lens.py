import numpy as np
from typing import Optional, Tuple, Any
from astropy import units as u

from ...core.component import Component, OpticalComponent
from ...core.layer import Layer, OpticalLayer
from ...core.pipeline import Pipeline
from ...core.wavefront import Wavefront, WavefrontArray


class Lens(OpticalLayer):
    """
    Thin lens element applying a quadratic phase to the pupil plane field.

    A paraxial thin lens introduces a phase term corresponding to a quadratic
    optical path difference, which in Fourier optics produces focusing.

    The phase imparted by a thin lens of focal length $f$ is given by:

    $$ \phi(x, y) = -\frac{k}{2f} (x^2 + y^2) $$

    where $k = \frac{2\pi}{\lambda}$ is the wavenumber and $(x, y)$ are pupil
    plane coordinates in meters.

    Parameters
    ----------
    focal_length : astropy.Quantity
        Lens focal length (meters).
    center : tuple of astropy.Quantity, optional
        Center offset in the pupil plane (meters) as `(x0, y0)`. Default `(0 m, 0 m)`.
    name : str, optional
        Descriptive name.

    Notes
    -----
    - Coordinates are derived from the wavefront's `pixel_scale` (meters per pixel).
      If `pixel_scale` is not set, a default of `1 m/pixel` is used.
    - For `WavefrontArray`, the phase is applied independently to each channel.
    - This element operates in the pupil plane; propagation to the focal plane
      should be performed downstream (e.g., by a detector or dedicated propagator).
    """

    def __init__(self, focal_length: u.Quantity, center: Tuple[u.Quantity, u.Quantity] = (0 * u.m, 0 * u.m), name: Optional[str] = None):
        super().__init__(name=name or "Lens")
        self.focal_length_m = float(focal_length.to(u.m).value)
        self.center_m = (float(center[0].to(u.m).value), float(center[1].to(u.m).value))

    def _apply_to_wavefront(self, wf: Wavefront) -> Wavefront:
        # Determine pixel scale (meters per pixel)
        scale_m = float(wf.pixel_scale.to(u.m).value) if hasattr(wf, 'pixel_scale') and hasattr(wf.pixel_scale, 'to') else 1.0

        if wf.ndim == 3:
            h, w = wf.shape[1], wf.shape[2]
        else:
            h, w = wf.shape
            
        # Coordinate grids centered at zero
        y = (np.arange(h) - (h - 1) / 2.0) * scale_m
        x = (np.arange(w) - (w - 1) / 2.0) * scale_m
        X, Y = np.meshgrid(x, y)

        # Apply center offset
        X = X - self.center_m[0]
        Y = Y - self.center_m[1]

        # Wavenumber k = 2*pi / lambda
        lam_m = float(wf.wavelength.to(u.m).value)
        k = 2.0 * np.pi / lam_m

        # Thin lens phase: phi = -(k/(2f)) * (x^2 + y^2)
        quad = X * X + Y * Y
        phi = -(k / (2.0 * self.focal_length_m)) * quad

        wf[:] = wf * np.exp(1j * phi).astype(wf.dtype)
        # Record focal length for downstream propagation
        wf._last_focal_length_m = self.focal_length_m
        return wf

    def process(self, wavefront: Any, context: Optional['Context'] = None) -> Any:
        """
        Apply the thin lens phase to a `Wavefront` or each channel in a `WavefrontArray`.

        Parameters
        ----------
        wavefront : Wavefront or WavefrontArray
            Input wavefront(s) in the pupil plane.

        Returns
        -------
        Wavefront or WavefrontArray
            Wavefront(s) with lens phase applied.
        """
        if wavefront is None:
            return None

        # Handle WavefrontArray
        if isinstance(wavefront, WavefrontArray):
            out = wavefront.copy()
            for i in range(len(out)):
                out[i] = self._apply_to_wavefront(out[i])
            return out

        # Single wavefront
        return self._apply_to_wavefront(wavefront)


def test_lens_basic():
    """Basic validation for Lens element."""
    lam = 550e-9 * u.m
    wf = Wavefront(wavelength=lam, size=128)
    # Set pixel scale to a reasonable value (placeholder)
    wf.pixel_scale = 0.01 * u.m

    lens = Lens(focal_length=10 * u.m)
    wf_out = lens.process(wf)

    assert wf_out.shape == (128, 128)
    # Phase should not be uniformly zero
    phase = np.angle(wf_out)
    assert np.std(phase) > 0.0

    print("✓ Lens basic phase application")


if __name__ == "__main__":
    test_lens_basic()
    print("All Lens tests passed.")
