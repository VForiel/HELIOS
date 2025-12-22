"""Adaptive Optics component."""
import numpy as np
import math
from astropy import units as u
from typing import Tuple, Optional, Dict
from ...core.pipeline import OpticalComponent
from ...core.simulation import Wavefront


class AdaptiveOptics(OpticalComponent):
    """Adaptive optics layer applying Zernike-based correction.

    Parameters
    ----------
    coeffs : dict, optional
        Mapping from (n,m) -> coefficient in radians. n >= 0, m integer with abs(m)<=n 
        and (n-abs(m)) even. Example: {(1,1): 0.1} for Zernike n=1,m=1.
    normalize : bool, optional
        Whether to evaluate Zernikes on unit pupil mapped to array size. Default: True
    name : str, optional
        Name of the AO system for identification in diagrams
    """
    def __init__(self, coeffs: Optional[dict] = None, normalize: bool = True, 
                 name: Optional[str] = None):
        super().__init__(name=name or "AdaptiveOptics")
        self.coeffs = coeffs or {}
        self.normalize = normalize

    @staticmethod
    def noll_to_nm(j: int) -> Tuple[int, int]:
        """Convert Noll index (1-based) to Zernike (n,m).

        This uses the standard Noll ordering. Returns (n,m).
        """
        if j < 1:
            raise ValueError("Noll index must be >= 1")
        # Noll indexing: j=1 -> (0,0); j=2 -> (1,-1); j=3 -> (1,1); j=4 -> (2,-2) ...
        # We'll compute by enumerating until reach index j.
        count = 0
        n = 0
        while True:
            for m in range(-n, n + 1, 2):
                count += 1
                if count == j:
                    return (n, m)
            n += 1

    def _radial_polynomial(self, n: int, m: int, r: np.ndarray) -> np.ndarray:
        m = abs(m)
        if (n - m) % 2 != 0:
            return np.zeros_like(r)
        R = np.zeros_like(r)
        kmax = (n - m) // 2
        for k in range(kmax + 1):
            num = (-1) ** k * math.factorial(n - k)
            den = math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
            R += num / den * r ** (n - 2 * k)
        return R

    def _zernike_nm(self, n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # m may be negative: negative -> sin component
        if m == 0:
            R = self._radial_polynomial(n, 0, rho)
            return R
        elif m > 0:
            R = self._radial_polynomial(n, m, rho)
            return R * np.cos(m * theta)
        else:
            R = self._radial_polynomial(n, -m, rho)
            return R * np.sin((-m) * theta)

    def process(self, wavefront: Wavefront) -> Wavefront:
        try:
            N = wavefront.npix
        except Exception:
            # Fallback if npix not available
            N = wavefront.shape[-1]

        # coordinates normalized to unit disk
        ys = np.linspace(-1.0, 1.0, N)
        xs = ys.copy()
        xg, yg = np.meshgrid(xs, ys)
        rho = np.hypot(xg, yg)
        theta = np.arctan2(yg, xg)
        mask = rho <= 1.0

        # build AO correction phase
        phase = np.zeros((N, N), dtype=float)
        # allow coeff keys to be either (n,m) tuples or Noll integer indices
        items = []
        for k, coeff in self.coeffs.items():
            if isinstance(k, int):
                nm = self.noll_to_nm(k)
            else:
                nm = tuple(k)
            items.append((nm, coeff))

        for (n, m), coeff in items:
            c = float(u.Quantity(coeff, u.rad).to(u.rad).value)
            Z = self._zernike_nm(n, m, rho, theta)
            phase += c * Z

        # apply only inside pupil (unit disk)
        phase = phase * mask
        # AO subtracts estimated phase (apply negative phase)
        wavefront[:] = wavefront * np.exp(-1j * phase).astype(wavefront.dtype)
        return wavefront
