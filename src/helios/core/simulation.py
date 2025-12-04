import numpy as np
from astropy import units as u
from typing import Optional, List, Tuple, Callable
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm

def _get_smart_extent(shape: Tuple[int, ...], pixel_scale: u.Quantity):
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
    if len(shape) == 3:
        H, W = shape[1], shape[2]
    else:
        H, W = shape[0], shape[1]
        
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
    samples : int, optional
        Number of spatial samples (wavefronts) to simulate. Default: 1.
        If > 1, field shape is (samples, size, size).
    
    Attributes
    ----------
    wavelength : Quantity
        Wavelength of the electromagnetic radiation
    field : ndarray of complex128
        Complex amplitude array representing the electric field.
        Shape is (samples, size, size). Amplitude = abs(field), phase = angle(field)
    pixel_scale : Quantity
        Physical size per pixel in meters (for pupil plane) or angular
        size per pixel (for image plane)
    source_directions : Quantity, optional
        (M, 2) array of source directions (theta_x, theta_y) in radians.
    
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
    def __init__(self, wavelength: u.Quantity, size: int, samples: int = 1):
        self.wavelength = wavelength
        self.field = np.ones((samples, size, size), dtype=np.complex128)
        self.pixel_scale = 1.0 * u.m # Placeholder
        self.max_modes: Optional[int] = None  # None for free-space, int for guided modes
        # Last optical focal length encountered (meters), set by lens-like elements
        self._last_focal_length_m: Optional[float] = None
        self.source_directions: Optional[u.Quantity] = None # (M, 2) angles

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
             show: bool = True, log_scale: bool = True, stack_method: Optional[Callable] = None):
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
        stack_method : callable, optional
            Function to aggregate samples (e.g., np.mean). If None, plots each sample independently.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : list of matplotlib.axes.Axes
            The axes objects.
        """
        # Handle stacking
        if stack_method is not None:
            # Apply stack method to amplitude and phase separately
            # Note: stacking complex field directly might lead to cancellation (interference)
            # User requested "moyenne des amplitudes et phases"
            amp_to_plot = stack_method(np.abs(self.field), axis=0)
            phase_to_plot = stack_method(np.angle(self.field), axis=0)
            
            # For log amplitude, compute log of stacked amplitude
            log_amp_to_plot = np.log10(amp_to_plot + 1e-12)
            
            # Treat as single plot
            fields_to_plot = [(amp_to_plot, phase_to_plot, log_amp_to_plot, "Stacked")]
        else:
            # Plot each sample independently with progress display
            fields_to_plot = []
            n_samples = self.field.shape[0]
            for i in tqdm(range(n_samples), desc="Stacking samples for plot", unit="sample", total=n_samples):
                amp = np.abs(self.field[i])
                phase = np.angle(self.field[i])
                log_amp = np.log10(amp + 1e-12)
                fields_to_plot.append((amp, phase, log_amp, f"Sample {i+1}"))
        
        # Determine layout
        n_plots = len(fields_to_plot)
        ncols = 3 if log_scale else 2
        nrows = n_plots
        
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)
            
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        # Build extent from pixel scale if available
        extent, x_label, y_label = _get_smart_extent(self.field.shape, self.pixel_scale)

        for i, (amp, phase, log_amp, label_suffix) in enumerate(fields_to_plot):
            # Amplitude
            ax1 = axes[i, 0]
            im1 = ax1.imshow(amp, cmap='inferno', origin='lower', extent=extent)
            ax1.set_title(f"Amplitude ({label_suffix})")
            ax1.set_xlabel(x_label)
            ax1.set_ylabel(y_label)
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Amplitude")

            current_col = 1
            # Log Amplitude (optional)
            if log_scale:
                ax_log = axes[i, current_col]
                im_log = ax_log.imshow(log_amp, cmap='inferno', origin='lower', extent=extent)
                ax_log.set_title(f"Log Amplitude ({label_suffix})")
                ax_log.set_xlabel(x_label)
                ax_log.set_ylabel(y_label)
                cbar_log = plt.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
                cbar_log.set_label("Log Amplitude")
                current_col += 1

            # Phase
            ax2 = axes[i, current_col]
            im2 = ax2.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi, origin='lower', extent=extent)
            ax2.set_title(f"Phase ({label_suffix})")
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
        # Apply along last two axes (spatial)
        self.field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.field, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
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
        
    def plot(self, title: Optional[str] = None, show: bool = True, log_scale: bool = True, stack_method: Optional[Callable] = None):
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
        stack_method : callable, optional
            Function to aggregate samples (e.g., np.mean). If None, plots each sample independently.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : ndarray of Axes
            The axes objects (rows x N columns).
        """
        n_channels = len(self.wavefronts)
        if n_channels == 0:
            return None, None
            
        # Determine number of samples per channel (assume all same)
        n_samples = self.wavefronts[0].field.shape[0]
        
        if stack_method is not None:
            n_rows_per_sample = 3 if log_scale else 2
            total_rows = n_rows_per_sample
        else:
            n_rows_per_sample = 3 if log_scale else 2
            total_rows = n_samples * n_rows_per_sample
            
        fig_width = max(8, min(14, 4 * n_channels))
        fig_height = 4 * total_rows
        
        fig, axes = plt.subplots(total_rows, n_channels, figsize=(fig_width, fig_height), squeeze=False)

        for i, wf in enumerate(tqdm(self.wavefronts, desc="Plotting channels", unit="ch", total=len(self.wavefronts))):
            # Prepare data
            if stack_method is not None:
                amp = stack_method(np.abs(wf.field), axis=0)
                phase = stack_method(np.angle(wf.field), axis=0)
                log_amp = np.log10(amp + 1e-12)
                samples_to_plot = [(amp, phase, log_amp, "Stacked")]
            else:
                samples_to_plot = []
                for s in tqdm(range(n_samples), desc=f"Samples ch {i+1}", unit="sample", total=n_samples):
                    amp = np.abs(wf.field[s])
                    phase = np.angle(wf.field[s])
                    log_amp = np.log10(amp + 1e-12)
                    samples_to_plot.append((amp, phase, log_amp, f"Sample {s+1}"))
            
            extent, xlabel, ylabel = _get_smart_extent(wf.field.shape, wf.pixel_scale)
            
            for s_idx, (amp, phase, log_amp, label_suffix) in enumerate(samples_to_plot):
                row_offset = s_idx * n_rows_per_sample
                
                # Amplitude
                ax_amp = axes[row_offset, i]
                im_amp = ax_amp.imshow(amp, origin='lower', cmap='inferno', extent=extent)
                ax_amp.set_title(f"Ch {i+1} Amp ({label_suffix})")
                ax_amp.set_xlabel(xlabel)
                ax_amp.set_ylabel(ylabel)
                cb_amp = plt.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
                cb_amp.set_label('Amplitude')
                
                current_row = row_offset + 1
                if log_scale:
                    # Log Amplitude
                    ax_log = axes[current_row, i]
                    im_log = ax_log.imshow(log_amp, origin='lower', cmap='inferno', extent=extent)
                    ax_log.set_title(f"Ch {i+1} Log Amp ({label_suffix})")
                    ax_log.set_xlabel(xlabel)
                    ax_log.set_ylabel(ylabel)
                    cb_log = plt.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
                    cb_log.set_label('Log Amplitude')
                    current_row += 1
                
                # Phase
                ax_phase = axes[current_row, i]
                im_phase = ax_phase.imshow(phase, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi, extent=extent)
                ax_phase.set_title(f"Ch {i+1} Phase ({label_suffix})")
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
        for wf in tqdm(self.wavefronts, desc="Propagating wavefronts", unit="wf", total=len(self.wavefronts)):
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
