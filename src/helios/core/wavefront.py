import numpy as np
import warnings
from typing import Optional, Tuple
import matplotlib.pyplot as plt

from astropy import units as u

from ..utils.plotting import get_smart_extent, format_coord

class Wavefront(u.Quantity):
    """
    Represents a monochromatic wavefront from a single coherent source.
    
    A wavefront describes the spatial distribution of light at a given wavelength
    from a single point source. The complex field contains both amplitude and phase
    information, enabling simulation of interference, diffraction, and aberrations.
    """
    def __new__(cls, wavelength: u.Quantity = 550*u.nm, size: u.Quantity = 1*u.m,
                 npix: int = 256,
                 value: Optional[np.ndarray] = None,
                 unit: u.Unit = u.dimensionless_unscaled,
                 dtype=np.complex128, copy=True, **kwargs):
        
        if value is None:
            shape = (int(npix), int(npix))
            value = np.ones(shape, dtype=dtype)
        else:
            if isinstance(value, u.Quantity):
                unit = value.unit
                value = value.value
            
            value = np.asanyarray(value)
            if value.ndim != 2:
                raise ValueError(f"Wavefront value must be 2D (npix × npix), got shape {value.shape}")
            
            if npix is not None and npix != value.shape[-1]:
                 warnings.warn(f"Provided npix={npix} does not match value shape {value.shape}. Using value shape.")

        obj = super().__new__(cls, value, unit=unit, dtype=dtype, copy=copy)
        return obj

    def __init__(self, wavelength: u.Quantity = 550*u.nm, size: u.Quantity = 1*u.m,
                 npix: int = 256,
                 value: Optional[np.ndarray] = None,
                 unit: u.Unit = u.dimensionless_unscaled,
                 dtype=np.complex128, copy=True, **kwargs):
        
        self.wavelength = wavelength
        self.width = size
        
        self.npix = self.shape[0] if self.ndim == 2 else self.shape[-1]
        
        size_q = self.width if isinstance(self.width, u.Quantity) else (self.width * u.m)
        self.pixel_scale = (size_q / self.npix).to(u.m)
        
        self.pixel_angle = None
        self.history = []

    def __array_finalize__(self, obj):
        if obj is None: return
        super().__array_finalize__(obj)
        
        self.wavelength = getattr(obj, 'wavelength', 550*u.nm)
        self.width = getattr(obj, 'width', 1*u.m)
        self.pixel_scale = getattr(obj, 'pixel_scale', None)
        self.pixel_angle = getattr(obj, 'pixel_angle', None)
        self.history = getattr(obj, 'history', [])
        
        if self.ndim == 2:
            self.npix = self.shape[0]
            
        if self.pixel_scale is not None and self.ndim == 2:
             self.width = self.npix * self.pixel_scale

    def copy(self) -> 'Wavefront':
        """Return a deep copy of the wavefront."""
        new_obj = super().copy()
        new_obj.history = list(self.history)
        return new_obj

    @property
    def amplitude(self):
        return np.abs(self)

    @property
    def intensity(self):
        return np.abs(self)**2

    @property
    def phase(self):
        return np.angle(self)

    @property
    def integrated_intensity(self):
        """Total energy (sum of intensity * pixel_area)."""
        # Note: Depending on unit normalization, this might need pixel_scale**2 factor
        # For pure numerical comparison:
        return np.sum(self.intensity)
    
    def coordinates(self) -> Tuple[u.Quantity, u.Quantity]:
        """Return (y, x) coordinate arrays for the wavefront grid."""
        h, w = self.shape
        scale = self.pixel_scale
        
        y_idx = np.arange(h) - (h - 1) / 2.0
        x_idx = np.arange(w) - (w - 1) / 2.0
        
        X_idx, Y_idx = np.meshgrid(x_idx, y_idx)
        return Y_idx * scale, X_idx * scale

    def plot(self, title: Optional[str] = None, figsize: Optional[tuple] = None, 
             show: bool = True, log_scale: bool = True, fov: Optional[u.Quantity] = None):
        """Basic plotting utility."""
        fig, axes = plt.subplots(1, 3 if log_scale else 2, figsize=figsize or (12, 4))
        
        extent, x_label, y_label = get_smart_extent(self.shape, self.pixel_scale)
        
        # Intensity
        im0 = axes[0].imshow(self.intensity, cmap='inferno', origin='lower', extent=extent)
        axes[0].set_title("Intensity")
        plt.colorbar(im0, ax=axes[0])
        
        # Phase
        im1 = axes[1].imshow(self.phase, cmap='twilight', vmin=-np.pi, vmax=np.pi, origin='lower', extent=extent)
        axes[1].set_title("Phase")
        plt.colorbar(im1, ax=axes[1])
        
        if log_scale:
            im2 = axes[2].imshow(np.log10(self.intensity + 1e-12), cmap='inferno', origin='lower', extent=extent)
            axes[2].set_title("Log Intensity")
            plt.colorbar(im2, ax=axes[2])
            
        if title: fig.suptitle(title)
        plt.tight_layout()
        if show: plt.show()
        return fig, axes

    def propagate(self, distance: u.Quantity, output_size: Optional[u.Quantity] = None, 
                  output_npix: Optional[int] = None, focal_length: Optional[u.Quantity] = None,
                  regime: Optional[str] = None) -> 'Wavefront':
        """
        Propagate the wavefront from pupil plane to detector plane.
        """
        # Lazy import to avoid circular dependencies
        try:
            from ..sim import propagation
        except ImportError:
            import helios.sim.propagation as propagation

        # Physical parameters in SI units
        d = distance.to(u.m).value
        wavelength = self.wavelength.to(u.m).value
        D = self.width.to(u.m).value
        N_in = self.npix
        dx_in = D / N_in
        
        # Initialize the field to be propagated (copy to avoid modifying self)
        field_in = self.value.copy()
        
        # --- 1. Apply Thin Lens Phase (if applicable) ---
        if focal_length is not None and regime != 'fraunhofer':
            f = focal_length.to(u.m).value
            k = 2 * np.pi / wavelength
            
            # Coordinate grid
            x = (np.arange(N_in) - N_in / 2) * dx_in
            X, Y = np.meshgrid(x, x)
            R2 = X**2 + Y**2
            
            # Lens phase (converging lens has negative phase)
            lens_phase = np.exp(-1j * k * R2 / (2 * f))
            
            # Correction: Use the initialized field_in
            field_in = field_in * lens_phase
            self.history.append(f"Applied Thin Lens Phase (f={f} m)")

        # --- 2. Determine Output Parameters ---
        L_fresnel = (wavelength * abs(d) * N_in) / D if abs(d) > 1e-9 else D
        
        if output_size is not None:
            L_target = output_size.to(u.m).value
        else:
             # Heuristic: if focusing, target Fresnel scale; else keep input scale
             if focal_length is not None and np.isclose(d, focal_length.to(u.m).value, rtol=0.1):
                 L_target = L_fresnel
             else:
                 L_target = D
                 
        L_out = L_target
        N_out = output_npix if output_npix is not None else N_in
        
        # --- 3. Regime Selection (Auto) ---
        if regime is None or regime.lower() == 'auto':
            z_min_fresnel = (D**2) / (N_in * wavelength)
            z_max_asm = (N_in * dx_in**2) / wavelength
            
            matches_fresnel_geom = np.isclose(L_target, L_fresnel, rtol=1e-2)
            matches_asm_geom = np.isclose(L_target, D, rtol=1e-2)
            
            if matches_fresnel_geom and abs(d) > z_min_fresnel:
                 regime = 'fresnel'
            elif matches_asm_geom and abs(d) < z_max_asm:
                 regime = 'asm'
            else:
                 # Fallback for zooms or non-standard distances
                 regime = 'scasm'
        
        regime = regime.lower()
        if regime == 'fresnel_fft': regime = 'fresnel'

        # --- 4. Dispatch Propagation ---
        if regime == 'fraunhofer':
             field_out = propagation.fraunhofer(field_in, D, wavelength, d, Lf=L_out, Nf=N_out)
        elif regime == 'fresnel':
            field_out = propagation.fresnel(field_in, D, wavelength, d, Lf=L_out, Nf=N_out)
        elif regime == 'asm':
            field_out = propagation.asm(field_in, D, wavelength, d, Lf=L_out, Nf=N_out)
        elif regime == 'scasm':
            field_out = propagation.scasm(field_in, D, wavelength, d, Lf=L_out, Nf=N_out)
        elif regime == 'rs_direct':
            field_out = propagation.rs_direct(field_in, D, wavelength, d, Lf=L_out, Nf=N_out)
        elif regime == 'fresnel_custom':
             field_out = propagation.fresnel_custom(field_in, D, wavelength, d, Lf=L_out, Nf=N_out)
        else:
             raise ValueError(f"Unknown propagation regime: {regime}")

        # Create output object
        wf_out = Wavefront(
            wavelength=self.wavelength,
            size=L_out * u.m,
            npix=N_out,
            value=field_out
        )
        
        wf_out.history = self.history.copy()
        wf_out.history.append(f"Propagated distance={distance}, method={regime}")
        
        return wf_out



def test_wavefront_init():
    wf = Wavefront(wavelength=600*u.nm, size=128)
    assert wf.shape == (1, 128, 128)
    assert wf.wavelength == 600 * u.nm

if __name__ == "__main__":
    test_wavefront_init()
    print("Simulation tests passed.")
