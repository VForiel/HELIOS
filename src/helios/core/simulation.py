import numpy as np
from astropy import units as u
import warnings
from typing import Optional, List, Tuple, Callable, Union
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

def _format_coord(coord: Union[u.Quantity, Tuple, List]) -> str:
    """
    Format a coordinate (tuple or Quantity) into a readable string with integer values if possible.
    Tries to find a unit (deg, arcmin, arcsec, mas, uas) where values are in [0, 999].
    """
    if isinstance(coord, (tuple, list)):
        # Check if elements are quantities
        if all(isinstance(c, u.Quantity) for c in coord):
            # Convert to array quantity
            try:
                coord = u.Quantity(coord)
            except:
                pass # Mixed units?
        else:
            # Plain numbers, assume radians if small? Or just print as is.
            # User said "si c'est des tuples ... affiche les en int en convertissant"
            # If plain floats, we don't know unit. Just format nicely.
            try:
                return f"({coord[0]:.2e}, {coord[1]:.2e})"
            except:
                return str(coord)

    if isinstance(coord, u.Quantity):
        # Flatten if needed
        vals = coord.flatten()
        if vals.size != 2:
            return str(coord)
        
        # Try units from smallest to largest to find best fit
        # Default for 0 is mas
        if np.max(np.abs(vals.value)) == 0:
             return "0, 0 mas"

        for unit in [u.uas, u.mas, u.arcsec, u.arcmin, u.deg]:
            try:
                v = vals.to(unit).value
                max_val = np.max(np.abs(v))
                
                if 0.1 <= max_val < 1000:
                    # Good range. Format as int if close to int, else float
                    # User asked for "affiche les en int"
                    return f"{int(round(v[0]))}, {int(round(v[1]))} {unit}"
            except:
                continue
        
        # Fallback
        return f"{vals[0].value:.2e}, {vals[1].value:.2e} {vals.unit}"

    return str(coord)

class Wavefront:
    """
    Represents the electromagnetic field (complex amplitude).
    
    A wavefront describes the spatial distribution of light at a given wavelength.
    The complex field contains both amplitude and phase information, enabling
    simulation of interference, diffraction, and aberrations.
    
    Parameters
    ----------
    wavelength : Quantity, optional
        Wavelength of the light (e.g., 550*u.nm). Default: 550 nm.
    size : Quantity, optional
        Physical width of the wavefront (e.g., 1*u.m). Default: 1 m.
    npix : int, optional
        Number of pixels along one dimension. Required if value is None.
    nsource : int, optional
        Number of incoherent wavefronts (sources). Required if value is None.
    value : ndarray, optional
        Complex field array of shape (nsource, npix, npix). If provided, overrides npix/nsource.
    
    Attributes
    ----------
    wavelength : Quantity
        Wavelength of the electromagnetic radiation
    field : ndarray of complex128
        Complex amplitude array representing the electric field.
        Shape is (nsource, npix, npix).
    pixel_scale : Quantity
        Physical size per pixel.
    source_directions : Quantity, optional
        (M, 2) array of source directions (theta_x, theta_y) in radians.
    
    Examples
    --------
    Create a wavefront and apply a phase aberration:
    
    >>> import numpy as np
    >>> from astropy import units as u
    >>> 
    >>> wf = Wavefront(wavelength=550*u.nm, size=1*u.m, npix=256, nsource=1)
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
    def __init__(self, wavelength: u.Quantity = 550*u.nm, size: u.Quantity = 1*u.m,
                 npix: int = 256, nsource: Optional[int] = 1,
                 value: Optional[np.ndarray] = None):

        self.wavelength = wavelength
        self.size = size
        
        if value is not None:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            if value.ndim == 2:
                value = value[np.newaxis, :, :]
            
            self.field = value.astype(np.complex128)
            self.nsource, self.npix, _ = self.field.shape
            
            if npix is not None and npix != self.npix:
                warnings.warn(f"Provided npix={npix} does not match value shape {self.field.shape}. Using value shape.")
            if nsource is not None and nsource != self.nsource:
                warnings.warn(f"Provided nsource={nsource} does not match value shape {self.field.shape}. Using value shape.")
                
        else:
            # If value is None, we need npix and nsource
            if nsource is None:
                nsource = 1
            self.npix = int(npix)
            self.nsource = int(nsource)
            self.field = np.ones((self.nsource, self.npix, self.npix), dtype=np.complex128)

        # Calculate pixel scale (ensure Quantity)
        size_q = self.size if isinstance(self.size, u.Quantity) else (self.size * u.m)
        self.pixel_scale = (size_q / self.npix).to(u.m)
        
        self.max_modes: Optional[int] = None
        self._last_focal_length_m: Optional[float] = None
        self.source_directions: Optional[u.Quantity] = None
        self.sources = None

    def crop(self, new_size: u.Quantity, center: Tuple[float, float] = (0, 0)):
        """
        Crop the wavefront to a new physical size.
        
        Parameters
        ----------
        new_size : u.Quantity
            New width of the field.
        center : tuple
            (x, y) center offset in physical units (same as new_size).
        """
        # Ensure units match
        if not isinstance(new_size, u.Quantity):
             new_size = new_size * u.m 
             
        current_size_m = self.size.to(u.m).value
        new_size_m = new_size.to(u.m).value
        
        if new_size_m > current_size_m:
            warnings.warn("Cropping to a larger size than current wavefront. Padding with zeros.")
            
        pixel_scale_m = self.pixel_scale.to(u.m).value
        
        # Calculate new number of pixels
        new_npix = int(np.round(new_size_m / pixel_scale_m))
        
        # Calculate center offset in pixels
        # Development mode: enforce astropy.Quantity for physical inputs
        # Expect center coordinates as quantities; convert explicitly
        center_x_m = u.Quantity(center[0], u.m).to(u.m).value
        center_y_m = u.Quantity(center[1], u.m).to(u.m).value
        
        offset_x_pix = int(np.round(center_x_m / pixel_scale_m))
        offset_y_pix = int(np.round(center_y_m / pixel_scale_m))
        
        # Center of current array
        cx = self.npix // 2
        cy = self.npix // 2
        
        # Half width of new array
        hw = new_npix // 2
        
        start_x = cx - hw + offset_x_pix
        start_y = cy - hw + offset_y_pix
        
        end_x = start_x + new_npix
        end_y = start_y + new_npix
        
        # Handle boundary checks
        if start_x < 0 or start_y < 0 or end_x > self.npix or end_y > self.npix:
             warnings.warn("Crop region is out of bounds. Result may be smaller or empty.")
             start_x = max(0, start_x)
             start_y = max(0, start_y)
             end_x = min(self.npix, end_x)
             end_y = min(self.npix, end_y)
        
        # Support both 2D (single-source) and 3D (multi-source) fields
        if self.field.ndim == 3:
            self.field = self.field[:, start_y:end_y, start_x:end_x]
            self.npix = self.field.shape[1]
        else:
            self.field = self.field[start_y:end_y, start_x:end_x]
            self.npix = self.field.shape[0]
        self.size = new_size

    def adapt(self, size: u.Quantity, magnify: Optional[bool] = None, npix: Optional[int] = None) -> 'Wavefront':
        """
        Adapt wavefront to match an optical element's physical size.
        
        This method adjusts the wavefront's metadata (size, pixel_scale) and optionally
        resamples the field to match a target optical element. It is designed to handle
        size mismatches between propagating wavefronts and optical components.
        
        Parameters
        ----------
        size : u.Quantity
            Target physical size (diameter/width) to adapt to.
        magnify : bool, optional
            If True, resize metadata without cropping (changes pixel_scale).
            If False, crop the wavefront to the target size.
            If None (default), auto-detect: magnify if sizes don't match, otherwise crop.
        npix : int, optional
            Target number of pixels for the adapted wavefront.
            If provided, resamples the field to this resolution.
            If None, keeps the current pixel count.
            
        Returns
        -------
        Wavefront
            A new wavefront adapted to the target size and resolution.
            
        Notes
        -----
        - **Magnify mode**: Adjusts `size` and `pixel_scale` without changing the field array.
          This is useful when the wavefront data is correct but metadata needs updating.
        - **Crop mode**: Physically crops the field to the target size using `crop()`.
        - **Resampling**: If `npix` is specified, uses scipy.ndimage.zoom to resample the field.
          This is useful for upscaling/downscaling the wavefront to match detector resolution.
          
        Examples
        --------
        Adapt a 2m wavefront to a 1m pupil:
        
        >>> wf = Wavefront(size=2*u.m, npix=512)
        >>> wf_adapted = wf.adapt(size=1*u.m, magnify=False)  # Crops to 1m
        
        Rescale a wavefront to 256 pixels:
        
        >>> wf_lowres = wf.adapt(size=wf.size, npix=256)  # Downsamples to 256x256
        """
        from scipy.ndimage import zoom
        
        # Create a copy to avoid modifying the original
        wf = self.copy()
        
        # Auto-magnify logic
        sizes_match = np.isclose(size.to(u.m).value, wf.size.to(u.m).value, rtol=1e-5)
        
        if magnify is None:
            if not sizes_match:
                warnings.warn(f"Wavefront size ({wf.size}) does not match target size ({size}). "
                              f"Resizing wavefront metadata to match (magnify=True).")
                magnify = True
            else:
                magnify = False
        
        # Apply size adaptation
        if magnify:
            wf.size = size
            wf.pixel_scale = (size / wf.npix).to(u.m)
        else:
            wf.crop(new_size=size, center=(0*u.m, 0*u.m))
        
        # Apply resampling if npix is specified
        if npix is not None and npix != wf.npix:
            zoom_factor = npix / wf.npix
            
            # Resample the field array
            if wf.field.ndim == 3:
                # Multi-source: zoom each source independently
                new_field = np.zeros((wf.nsource, npix, npix), dtype=np.complex128)
                for i in range(wf.nsource):
                    # Zoom real and imaginary parts separately
                    real_part = zoom(wf.field[i].real, zoom_factor, order=3)
                    imag_part = zoom(wf.field[i].imag, zoom_factor, order=3)
                    new_field[i] = real_part + 1j * imag_part
            else:
                # Single source 2D field
                real_part = zoom(wf.field.real, zoom_factor, order=3)
                imag_part = zoom(wf.field.imag, zoom_factor, order=3)
                new_field = real_part + 1j * imag_part
            
            wf.field = new_field
            wf.npix = npix
            wf.pixel_scale = (wf.size / npix).to(u.m)
        
        return wf

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
        if self.sources is not None:
            new_wf.sources = copy.deepcopy(self.sources)
        return new_wf

    def plot(self, title: Optional[str] = None, figsize: Optional[tuple] = None, 
             show: bool = True, log_scale: bool = True, stack_method: Optional[Callable] = None,
             max_plots: int = 5, fov: Optional[u.Quantity] = None):
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
        max_plots : int, optional
            Maximum number of wavefronts to plot if stack_method is None. Default 5.
        fov : astropy.Quantity, optional
            Field of view to display (e.g., 2*u.arcsec). If None, shows the full array.
            
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
            intensity_to_plot = stack_method(np.abs(self.field)**2, axis=0)
            phase_to_plot = stack_method(np.angle(self.field), axis=0)
            
            # For log amplitude, compute log of stacked amplitude
            log_amp_to_plot = np.log10(amp_to_plot + 1e-12)
            log_intensity_to_plot = np.log10(intensity_to_plot + 1e-12)
            
            # Treat as single plot
            fields_to_plot = [(intensity_to_plot, log_intensity_to_plot, amp_to_plot, log_amp_to_plot, phase_to_plot, "Stacked")]
        else:
            # Plot each sample independently with progress display
            fields_to_plot = []
            n_samples = self.field.shape[0]
            
            n_to_plot = min(n_samples, max_plots)
            if n_samples > max_plots:
                print(f"Warning: Displaying only first {max_plots} of {n_samples} wavefronts.")
                
            for i in tqdm(range(n_to_plot), desc="Stacking samples for plot", unit="sample", total=n_to_plot):
                amp = np.abs(self.field[i])
                intensity = amp**2
                phase = np.angle(self.field[i])
                log_amp = np.log10(amp + 1e-12)
                log_intensity = np.log10(intensity + 1e-12)
                
                # Determine label
                label = f"Sample {i+1}"
                if self.sources is not None and i < len(self.sources):
                    src = self.sources[i]
                    if isinstance(src, str):
                        label = src
                    else:
                        label = _format_coord(src)
                
                fields_to_plot.append((intensity, log_intensity, amp, log_amp, phase, label))
        
        # Determine layout
        n_plots = len(fields_to_plot)
        ncols = 5 if log_scale else 3
        nrows = n_plots
        
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)
            
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        # Build extent from pixel scale if available
        extent, x_label, y_label = _get_smart_extent(self.field.shape, self.pixel_scale)

        for i, (intensity, log_intensity, amp, log_amp, phase, label_suffix) in enumerate(fields_to_plot):
            # Intensity
            ax_int = axes[i, 0]
            im_int = ax_int.imshow(intensity, cmap='inferno', origin='lower', extent=extent)
            ax_int.set_title(f"Intensity ({label_suffix})")
            ax_int.set_xlabel(x_label)
            ax_int.set_ylabel(y_label)
            cbar_int = plt.colorbar(im_int, ax=ax_int, fraction=0.046, pad=0.04)
            cbar_int.set_label("Intensity")
            
            current_col = 1
            # Log Intensity (optional)
            if log_scale:
                ax_log_int = axes[i, current_col]
                im_log_int = ax_log_int.imshow(log_intensity, cmap='inferno', origin='lower', extent=extent)
                ax_log_int.set_title(f"Log Intensity ({label_suffix})")
                ax_log_int.set_xlabel(x_label)
                ax_log_int.set_ylabel(y_label)
                cbar_log_int = plt.colorbar(im_log_int, ax=ax_log_int, fraction=0.046, pad=0.04)
                cbar_log_int.set_label("Log Intensity")
                current_col += 1

            # Amplitude
            ax1 = axes[i, current_col]
            im1 = ax1.imshow(amp, cmap='inferno', origin='lower', extent=extent)
            ax1.set_title(f"Amplitude ({label_suffix})")
            ax1.set_xlabel(x_label)
            ax1.set_ylabel(y_label)
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Amplitude")
            current_col += 1

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
            
            # Apply FOV if requested
            if fov is not None:
                # Convert FOV to same unit as extent
                # extent is [xmin, xmax, ymin, ymax] in unit 'unit' (from _get_smart_extent)
                # We need to know 'unit'. _get_smart_extent returns xlabel "x [unit]"
                # Let's parse it or re-derive it?
                # Easier: _get_smart_extent logic is simple.
                # Let's just use the fact that extent is centered.
                
                # Determine unit from xlabel
                unit_str = x_label.split('[')[-1].split(']')[0]
                try:
                    # Simple mapping for common units
                    unit_map = {'m': u.m, 'mm': u.mm, 'um': u.um, 'nm': u.nm, 
                                'rad': u.rad, 'deg': u.deg, 'arcmin': u.arcmin, 'arcsec': u.arcsec, 'mas': u.mas, 'uas': u.uas}
                    plot_unit = unit_map.get(unit_str, None)
                    
                    if plot_unit is not None:
                        limit = (fov / 2).to(plot_unit).value
                        for ax in axes[i, :]:
                            ax.set_xlim(-limit, limit)
                            ax.set_ylim(-limit, limit)
                except Exception as e:
                    print(f"Warning: Could not apply FOV: {e}")
        
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig, axes

    def propagate(self, distance: Optional[u.Quantity] = None, padding: int = 1):
        """
        Propagate the wavefront towards the focal (image) plane.

        If `distance` is None, attempt to use the last focal length set by an
        optical element (e.g., `Lens`). If no focal length is known, the
        wavefront is returned unchanged and a warning is emitted.

        Parameters
        ----------
        distance : astropy.Quantity, optional
            Propagation distance. If None, uses the stored focal length if available.
        padding : int, optional
            Zero-padding factor to increase sampling resolution in the focal plane.
            A value of 2 means the array size is doubled (padded with zeros),
            resulting in 2x finer resolution in the output. Default: 1 (no padding).

        Notes
        -----
        This implementation performs a Fraunhofer propagation (FFT-based):
        E_image = FFT{ E_pupil } with appropriate frequency centering.
        
        The pixel scale is updated according to:
        dx' = (lambda * f) / (N * dx)
        where N is the total grid size (including padding).
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

        # Apply padding if requested
        if padding > 1:
            # field shape is (samples, size, size)
            samples, h, w = self.field.shape
            new_h, new_w = h * padding, w * padding
            
            # Create new array with zeros
            new_field = np.zeros((samples, new_h, new_w), dtype=np.complex128)
            
            # Insert old field in the center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            new_field[:, start_h:start_h+h, start_w:start_w+w] = self.field
            
            self.field = new_field

        # Basic FFT-based Fraunhofer propagation to focal plane
        # Center -> FFT -> center
        # Apply along last two axes (spatial)
        self.field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.field, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # Update pixel scale
        # dx' = (lambda * f) / (N * dx)
        # N is the size of the array used for FFT (now potentially padded)
        N = self.field.shape[-1]
        
        # Ensure units are handled correctly
        # self.pixel_scale is Quantity [length]
        # self.wavelength is Quantity [length]
        # d_m is float [meters]
        
        # We need to be careful with units.
        # If pixel_scale is angular (e.g. rad), this formula is different.
        # But propagate usually goes from Pupil (m) to Image (m or rad).
        # Here we calculate the physical size in the focal plane (m).
        
        if self.pixel_scale.unit.is_equivalent(u.m):
            new_scale = (self.wavelength * (d_m * u.m)) / (N * self.pixel_scale)
            self.pixel_scale = new_scale.to(u.m)
        else:
            # If input is already angular? That's unusual for Fraunhofer from pupil.
            # Assume input is pupil plane in meters.
            pass
            
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
        
    def plot(self, title: Optional[str] = None, show: bool = True, log_scale: bool = True, 
             stack_method: Optional[Callable] = None, fov: Optional[u.Quantity] = None):
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
            # Stacked mode: 1 set of plots per channel (stacked over samples)
            # Layout: Channel 1, Channel 2, ...
            n_rows_per_item = 3 if log_scale else 2
            total_rows = n_channels * n_rows_per_item
            
            fig_width = max(8, min(14, 4 * (5 if log_scale else 3))) # Fixed width for Amp/Phase cols
            fig_height = 4 * total_rows
            
            fig, axes = plt.subplots(total_rows, 5 if log_scale else 3, figsize=(fig_width, fig_height), squeeze=False)
            
            for i, wf in enumerate(tqdm(self.wavefronts, desc="Plotting channels (stacked)", unit="ch")):
                amp = stack_method(np.abs(wf.field), axis=0)
                intensity = stack_method(np.abs(wf.field)**2, axis=0)
                phase = stack_method(np.angle(wf.field), axis=0)
                log_amp = np.log10(amp + 1e-12)
                log_intensity = np.log10(intensity + 1e-12)
                
                extent, xlabel, ylabel = _get_smart_extent(wf.field.shape, wf.pixel_scale)
                
                row_base = i * n_rows_per_item
                
                # Intensity
                ax_int = axes[row_base, 0]
                im_int = ax_int.imshow(intensity, origin='lower', cmap='inferno', extent=extent)
                ax_int.set_title(f"Ch {i+1} Int (Stacked)")
                ax_int.set_xlabel(xlabel)
                ax_int.set_ylabel(ylabel)
                cb_int = plt.colorbar(im_int, ax=ax_int, fraction=0.046, pad=0.04)
                cb_int.set_label('Intensity')
                
                current_col = 1
                # Log Intensity
                if log_scale:
                    ax_log_int = axes[row_base, current_col]
                    im_log_int = ax_log_int.imshow(log_intensity, origin='lower', cmap='inferno', extent=extent)
                    ax_log_int.set_title(f"Ch {i+1} Log Int (Stacked)")
                    ax_log_int.set_xlabel(xlabel)
                    ax_log_int.set_ylabel(ylabel)
                    cb_log_int = plt.colorbar(im_log_int, ax=ax_log_int, fraction=0.046, pad=0.04)
                    cb_log_int.set_label('Log Intensity')
                    current_col += 1

                # Amplitude
                ax_amp = axes[row_base, current_col]
                im_amp = ax_amp.imshow(amp, origin='lower', cmap='inferno', extent=extent)
                ax_amp.set_title(f"Ch {i+1} Amp (Stacked)")
                ax_amp.set_xlabel(xlabel)
                ax_amp.set_ylabel(ylabel)
                cb_amp = plt.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
                cb_amp.set_label('Amplitude')
                current_col += 1
                
                # Log Amplitude
                if log_scale:
                    ax_log = axes[row_base, current_col]
                    im_log = ax_log.imshow(log_amp, origin='lower', cmap='inferno', extent=extent)
                    ax_log.set_title(f"Ch {i+1} Log Amp (Stacked)")
                    ax_log.set_xlabel(xlabel)
                    ax_log.set_ylabel(ylabel)
                    cb_log = plt.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
                    cb_log.set_label('Log Amplitude')
                    current_col += 1

                # Phase
                ax_phase = axes[row_base, current_col]
                im_phase = ax_phase.imshow(phase, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi, extent=extent)
                ax_phase.set_title(f"Ch {i+1} Phase (Stacked)")
                ax_phase.set_xlabel(xlabel)
                ax_phase.set_ylabel(ylabel)
                cb_phase = plt.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
                cb_phase.set_label('Phase (rad)')
                
                # Apply FOV if requested
                if fov is not None:
                    unit_str = xlabel.split('[')[-1].split(']')[0]
                    try:
                        unit_map = {'m': u.m, 'mm': u.mm, 'um': u.um, 'nm': u.nm, 
                                    'rad': u.rad, 'deg': u.deg, 'arcmin': u.arcmin, 'arcsec': u.arcsec, 'mas': u.mas, 'uas': u.uas}
                        plot_unit = unit_map.get(unit_str, None)
                        
                        if plot_unit is not None:
                            limit = (fov / 2).to(plot_unit).value
                            for ax in axes[row_base, :]:
                                ax.set_xlim(-limit, limit)
                                ax.set_ylim(-limit, limit)
                    except Exception as e:
                        print(f"Warning: Could not apply FOV: {e}")
                    
        else:
            # Individual samples mode
            # Layout: Source 1 (Ch 1, Ch 2...), Source 2 (Ch 1, Ch 2...)
            n_rows_per_item = 1 # We put Amp/Phase side-by-side in one row? 
            # No, user wants "same logic as Wavefront.plot() with amplitude left and phase right"
            # In Wavefront.plot, it's 1 row per sample, with columns for Amp, Phase, LogAmp.
            # So here, for each (Source, Channel) pair, we want 1 row.
            
            total_rows = n_samples * n_channels
            # Columns: Amp, (LogAmp if enabled), Phase
            ncols = 3 if log_scale else 2
            
            fig_width = max(8, min(14, 4 * ncols))
            fig_height = 4 * total_rows
            
            fig, axes = plt.subplots(total_rows, ncols, figsize=(fig_width, fig_height), squeeze=False)
            
            row_idx = 0
            for s in tqdm(range(n_samples), desc="Plotting sources", unit="src"):
                for c, wf in enumerate(self.wavefronts):
                    # Get data
                    amp = np.abs(wf.field[s])
                    phase = np.angle(wf.field[s])
                    log_amp = np.log10(amp + 1e-12)
                    
                    # Determine label
                    label = f"Src {s+1} - Ch {c+1}"
                    if wf.sources is not None and s < len(wf.sources):
                        src = wf.sources[s]
                        if isinstance(src, str):
                            src_name = src
                        else:
                            src_name = _format_coord(src)
                        label = f"{src_name} - Ch {c+1}"
                    
                    extent, xlabel, ylabel = _get_smart_extent(wf.field.shape, wf.pixel_scale)
                    
                    # Amplitude (col 0)
                    ax_amp = axes[row_idx, 0]
                    im_amp = ax_amp.imshow(amp, origin='lower', cmap='inferno', extent=extent)
                    ax_amp.set_title(f"Amp ({label})")
                    ax_amp.set_xlabel(xlabel)
                    ax_amp.set_ylabel(ylabel)
                    cb_amp = plt.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
                    cb_amp.set_label('Amplitude')
                    
                    # Log Amplitude
                    if log_scale:
                        ax_log = axes[row_idx, 1]
                        im_log = ax_log.imshow(log_amp, origin='lower', cmap='inferno', extent=extent)
                        ax_log.set_title(f"Log Amp ({label})")
                        ax_log.set_xlabel(xlabel)
                        ax_log.set_ylabel(ylabel)
                        cb_log = plt.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
                        cb_log.set_label('Log Amplitude')
                        
                    # Phase
                    ax_phase = axes[row_idx, (2 if log_scale else 1)]
                    im_phase = ax_phase.imshow(phase, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi, extent=extent)
                    ax_phase.set_title(f"Phase ({label})")
                    ax_phase.set_xlabel(xlabel)
                    ax_phase.set_ylabel(ylabel)
                    cb_phase = plt.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
                    cb_phase.set_label('Phase (rad)')
                    
                    # Apply FOV if requested
                    if fov is not None:
                        unit_str = xlabel.split('[')[-1].split(']')[0]
                        try:
                            unit_map = {'m': u.m, 'mm': u.mm, 'um': u.um, 'nm': u.nm, 
                                        'rad': u.rad, 'deg': u.deg, 'arcmin': u.arcmin, 'arcsec': u.arcsec, 'mas': u.mas, 'uas': u.uas}
                            plot_unit = unit_map.get(unit_str, None)
                            
                            if plot_unit is not None:
                                limit = (fov / 2).to(plot_unit).value
                                for ax in axes[row_idx, :]:
                                    ax.set_xlim(-limit, limit)
                                    ax.set_ylim(-limit, limit)
                        except Exception as e:
                            print(f"Warning: Could not apply FOV: {e}")
                    
                    row_idx += 1

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        if show:
            plt.show()
            
        return fig, axes

    def propagate(self, distance: Optional[u.Quantity] = None, padding: int = 1) -> 'WavefrontArray':
        """
        Propagate all wavefronts in the array.

        Parameters
        ----------
        distance : astropy.Quantity, optional
            Propagation distance passed to each `Wavefront.propagate()`. If None,
            each wavefront uses its own stored focal length if available.
        padding : int, optional
            Zero-padding factor passed to `Wavefront.propagate()`. Default: 1.

        Returns
        -------
        WavefrontArray
            New array with propagated wavefronts. Locations are preserved.
        """
        propagated_wfs = []
        for wf in tqdm(self.wavefronts, desc="Propagating wavefronts", unit="wf", total=len(self.wavefronts)):
            propagated_wfs.append(wf.copy().propagate(distance, padding=padding))
        new_locs = list(self.locations) if self.locations is not None else None
        return WavefrontArray(propagated_wfs, locations=new_locs)

def test_wavefront_init():
    wf = Wavefront(wavelength=600*u.nm, size=128)
    assert wf.field.shape == (128, 128)
    assert wf.wavelength == 600 * u.nm

if __name__ == "__main__":
    test_wavefront_init()
    print("Simulation tests passed.")
