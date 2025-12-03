import numpy as np
from astropy import units as u
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import copy

def _get_smart_extent(shape: Tuple[int, int], pixel_scale: u.Quantity):
    """
    Determine plot extent and axis labels with appropriate units.
    
    Parameters
    ----------
    shape : tuple
        (height, width) of the array.
    pixel_scale : astropy.Quantity
        Physical or angular size of one pixel.
        
    Returns
    -------
    extent : list
        [xmin, xmax, ymin, ymax]
    xlabel : str
        Label for x axis with unit.
    ylabel : str
        Label for y axis with unit.
    """
    H, W = shape
    extent = None
    xlabel = 'x (pix)'
    ylabel = 'y (pix)'
    
    if pixel_scale is not None and isinstance(pixel_scale, u.Quantity):
        ps = pixel_scale
        # Determine best unit based on total field of view
        total_width = W * ps
        unit = ps.unit
        
        if unit.is_equivalent(u.m):
            if total_width < 100 * u.um:
                unit = u.um
            elif total_width < 1 * u.m:
                unit = u.mm
            else:
                unit = u.m
        elif unit.is_equivalent(u.rad):
            if total_width < 1 * u.arcsec:
                unit = u.mas
            elif total_width < 2 * u.deg:
                unit = u.arcsec
            else:
                unit = u.deg
                
        # Calculate extent
        half_x = (W / 2) * ps.to(unit).value
        half_y = (H / 2) * ps.to(unit).value
        extent = [-half_x, half_x, -half_y, half_y]
        xlabel = f"x [{unit}]"
        ylabel = f"y [{unit}]"
        
    return extent, xlabel, ylabel

class Wavefront:
    """
    Represents the electromagnetic field (complex amplitude).
    
    A wavefront describes the spatial distribution of light at a given wavelength.
    The complex field contains both amplitude and phase information, enabling
    simulation of interference, diffraction, and aberrations.
    
    Parameters
    ----------
    wavelength : Quantity
        Wavelength of the light (e.g., 550*u.nm, 1.6*u.um)
    size : int
        Number of pixels in the field array (creates size × size array)
    
    Attributes
    ----------
    wavelength : Quantity
        Wavelength of the electromagnetic radiation
    field : ndarray of complex128
        Complex amplitude array representing the electric field.
        Shape is (size, size). Amplitude = abs(field), phase = angle(field)
    pixel_scale : Quantity
        Physical size per pixel in meters (for pupil plane) or angular
        size per pixel (for image plane)
    
    Examples
    --------
    Create a wavefront and apply a phase aberration:
    
    >>> import numpy as np
    >>> from astropy import units as u
    >>> 
    >>> wf = Wavefront(wavelength=550*u.nm, size=256)
    >>> # Apply pupil amplitude
    >>> pupil = helios.Pupil.like('JWST')
    >>> wf.field = pupil.get_array(256).astype(np.complex128)
    >>> # Add phase aberration
    >>> phase = np.random.randn(256, 256) * 0.5  # radians
    >>> wf.field *= np.exp(1j * phase)
    >>> # Visualize
    >>> wf.plot(title="Aberrated Wavefront")
    
    Notes
    -----
    The field is typically initialized to uniform amplitude (ones) and then
    modified by layers to include pupil masks, phase aberrations, etc.
    
    The complex field enables coherent propagation:
    - Fourier transform for Fraunhofer diffraction
    - Fresnel propagation for arbitrary distances
    - Interference between multiple beams
    
    See Also
    --------
    Layer : Components that transform wavefronts
    """
    def __init__(self, wavelength: u.Quantity, size: int):
        self.wavelength = wavelength
        self.field = np.ones((size, size), dtype=np.complex128)
        self.pixel_scale = 1.0 * u.m # Placeholder
        self.max_modes: Optional[int] = None  # None for free-space, int for guided modes
        # Last optical focal length encountered (meters), set by lens-like elements
        self._last_focal_length_m: Optional[float] = None

    def copy(self) -> 'Wavefront':
        """
        Return a deep copy of the wavefront.
        
        Returns
        -------
        Wavefront
            A new Wavefront instance with independent field array.
        """
        new_wf = copy.copy(self)
        new_wf.field = self.field.copy()
        return new_wf

    def plot(self, title: Optional[str] = None, figsize: Optional[tuple] = None, 
             show: bool = True, log_scale: bool = True):
        """
        Plot the wavefront amplitude and phase side by side.
        
        Parameters
        ----------
        title : str, optional
            Super title for the figure.
        figsize : tuple, optional
            Figure size (width, height). Default (12, 5) or (18, 5) if log_scale=True.
        show : bool, optional
            If True, call plt.show(). Default True.
        log_scale : bool, optional
            If True, adds a third plot with log10(Amplitude). Default False.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : list of matplotlib.axes.Axes
            The axes objects.
        """
        ncols = 3 if log_scale else 2
        if figsize is None:
            figsize = (18, 5) if log_scale else (12, 5)
            
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1:
            axes = [axes]

        # Build extent from pixel scale if available
        extent, x_label, y_label = _get_smart_extent(self.field.shape, self.pixel_scale)

        # Amplitude
        ax1 = axes[0]
        im1 = ax1.imshow(np.abs(self.field), cmap='inferno', origin='lower', extent=extent)
        ax1.set_title("Amplitude")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Amplitude")

        # Log Amplitude (optional)
        current_ax_idx = 1
        if log_scale:
            ax_log = axes[current_ax_idx]
            # Add small epsilon to avoid log(0)
            log_amp = np.log10(np.abs(self.field) + 1e-12)
            im_log = ax_log.imshow(log_amp, cmap='inferno', origin='lower', extent=extent)
            ax_log.set_title("Log Amplitude")
            ax_log.set_xlabel(x_label)
            ax_log.set_ylabel(y_label)
            cbar_log = plt.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
            cbar_log.set_label("Log Amplitude")
            current_ax_idx += 1

        # Phase
        ax2 = axes[current_ax_idx]
        im2 = ax2.imshow(np.angle(self.field), cmap='twilight', vmin=-np.pi, vmax=np.pi, origin='lower', extent=extent)
        ax2.set_title("Phase")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Phase (rad)")
        
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig, axes

    def propagate(self, distance: Optional[u.Quantity] = None):
        """
        Propagate the wavefront towards the focal (image) plane.

        If `distance` is None, attempt to use the last focal length set by an
        optical element (e.g., `Lens`). If no focal length is known, the
        wavefront is returned unchanged and a warning is emitted.

        Parameters
        ----------
        distance : astropy.Quantity, optional
            Propagation distance. If None, uses the stored focal length if available.

        Notes
        -----
        This implementation performs a Fraunhofer propagation (FFT-based):
        E_image = FFT{ E_pupil } with appropriate frequency centering.
        It does not currently apply physical scaling factors to coordinates
        or amplitudes; this will be refined in future versions.
        """
        import warnings

        # Resolve propagation distance
        if distance is None:
            if self._last_focal_length_m is None:
                warnings.warn(
                    "Wavefront.propagate: No distance provided and no focal length known; returning input (0 m propagation).",
                    RuntimeWarning
                )
                return self
            else:
                # Use stored focal length
                d_m = float(self._last_focal_length_m)
        else:
            d_m = float(distance.to(u.m).value)

        # Basic FFT-based Fraunhofer propagation to focal plane
        # Center -> FFT -> center
        self.field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.field)))
        return self

class Simulation:
    """
    Helper class for running specific simulation types if needed.
    
    This class provides utilities for common simulation workflows and may
    be extended in the future for specialized observation modes.
    
    Notes
    -----
    Most simulations should use Context directly. This class is reserved
    for future specialized simulation types or batch processing workflows.
    
    See Also
    --------
    Context : Main simulation orchestrator
    """
    pass

class WavefrontArray:
    """
    Collection of wavefronts for parallel optical paths.
    
    Manages multiple Wavefront objects corresponding to different beams
    (e.g. in an interferometer or after beam splitting).
    
    Parameters
    ----------
    wavefronts : list of Wavefront, optional
        Initial list of wavefronts.
    locations : list of tuple, optional
        (x, y) positions of each wavefront center in the pupil plane (meters).
        Used for interferometric recombination.
    """
    def __init__(self, wavefronts: Optional[List[Wavefront]] = None, 
                 locations: Optional[List[Tuple[float, float]]] = None):
        self.wavefronts = wavefronts if wavefronts is not None else []
        self.locations = locations
        
        # Validate locations length
        if self.locations is not None and len(self.locations) != len(self.wavefronts):
            # If mismatch, warn or truncate? For now, just keep as is, but it might cause issues.
            pass

    def __getitem__(self, index: int) -> Wavefront:
        return self.wavefronts[index]
    
    def __setitem__(self, index: int, value: Wavefront):
        self.wavefronts[index] = value
        
    def __len__(self) -> int:
        return len(self.wavefronts)
        
    def __iter__(self):
        return iter(self.wavefronts)
        
    def append(self, wavefront: Wavefront, location: Optional[Tuple[float, float]] = None):
        self.wavefronts.append(wavefront)
        if self.locations is not None:
            if location is None:
                location = (0.0, 0.0)
            self.locations.append(location)
        elif location is not None:
            # Initialize locations with (0,0) for existing ones
            self.locations = [(0.0, 0.0)] * (len(self.wavefronts) - 1)
            self.locations.append(location)
        
    def copy(self) -> 'WavefrontArray':
        """Return a deep copy of the wavefront array."""
        new_locs = list(self.locations) if self.locations is not None else None
        return WavefrontArray([wf.copy() for wf in self.wavefronts], locations=new_locs)
        
    def plot(self, title: Optional[str] = None, show: bool = True, log_scale: bool = True):
        """
        Plot all wavefronts in the array (Amplitude and Phase).
        
        Creates a grid of subplots with 2 rows (Amplitude, Phase) and N columns
        (one for each wavefront). If log_scale is True, adds a row for Log Amplitude.
        
        Parameters
        ----------
        title : str, optional
            Super title for the figure.
        show : bool, optional
            If True, call plt.show(). Default True.
        log_scale : bool, optional
            If True, adds a row with log10(Amplitude). Default False.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : ndarray of Axes
            The axes objects (rows x N columns).
        """
        n = len(self.wavefronts)
        nrows = 3 if log_scale else 2
        fig_width = max(8, min(14, 4 * n))
        fig_height = 12 if log_scale else 8
        
        # rows (Amp, [LogAmp], Phase), N columns
        fig, axes = plt.subplots(nrows, n, figsize=(fig_width, fig_height))
        
        # Ensure axes is 2D array [row, col]
        if n == 1:
            axes = axes.reshape(nrows, 1)
        elif len(axes.shape) == 1:
            # Should not happen with >1 rows, but safety check
            axes = axes.reshape(nrows, n)

        for i, wf in enumerate(self.wavefronts):
            amp = np.abs(wf.field)
            phase = np.angle(wf.field)

            extent, xlabel, ylabel = _get_smart_extent(amp.shape, wf.pixel_scale)

            # Row 0: Amplitude
            ax_amp = axes[0, i]
            im_amp = ax_amp.imshow(amp, origin='lower', cmap='inferno', extent=extent)
            ax_amp.set_title(f"Ch {i+1} Amp")
            ax_amp.set_xlabel(xlabel)
            ax_amp.set_ylabel(ylabel)
            cb_amp = plt.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
            cb_amp.set_label('Amplitude')

            current_row = 1
            if log_scale:
                # Row 1: Log Amplitude
                ax_log = axes[current_row, i]
                log_amp = np.log10(amp + 1e-12)
                im_log = ax_log.imshow(log_amp, origin='lower', cmap='inferno', extent=extent)
                ax_log.set_title(f"Ch {i+1} Log Amp")
                ax_log.set_xlabel(xlabel)
                ax_log.set_ylabel(ylabel)
                cb_log = plt.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
                cb_log.set_label('Log Amplitude')
                current_row += 1

            # Row 1 or 2: Phase
            ax_phase = axes[current_row, i]
            im_phase = ax_phase.imshow(phase, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi, extent=extent)
            ax_phase.set_title(f"Ch {i+1} Phase")
            ax_phase.set_xlabel(xlabel)
            ax_phase.set_ylabel(ylabel)
            cb_phase = plt.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
            cb_phase.set_label('Phase (rad)')

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        if show:
            plt.show()
            
        return fig, axes

    def propagate(self, distance: Optional[u.Quantity] = None) -> 'WavefrontArray':
        """
        Propagate all wavefronts in the array.

        Parameters
        ----------
        distance : astropy.Quantity, optional
            Propagation distance passed to each `Wavefront.propagate()`. If None,
            each wavefront uses its own stored focal length if available.

        Returns
        -------
        WavefrontArray
            New array with propagated wavefronts. Locations are preserved.
        """
        propagated_wfs = []
        for wf in self.wavefronts:
            propagated_wfs.append(wf.copy().propagate(distance))
        new_locs = list(self.locations) if self.locations is not None else None
        return WavefrontArray(propagated_wfs, locations=new_locs)

def test_wavefront_init():
    wf = Wavefront(wavelength=600*u.nm, size=128)
    assert wf.field.shape == (128, 128)
    assert wf.wavelength == 600 * u.nm

if __name__ == "__main__":
    test_wavefront_init()
    print("Simulation tests passed.")
