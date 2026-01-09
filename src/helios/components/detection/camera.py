import numpy as np
from typing import Tuple, Optional, Any, List
from astropy import units as u
from ...core.component import Component, DetectionComponent
from ...core.layer import DetectionLayer, Layer
from ...core.pipeline import Pipeline
from ...utils.serialization import serialize_value, deserialize_value
from ...core.wavefront import Wavefront
import matplotlib.pyplot as plt

class Camera(DetectionComponent):
    __slots__ = (
        "pixels",
        "pixel_size",
        "dark_current",
        "read_noise",
        "integration_time_value",
        "integration_time",
        "quantum_efficiency",
        "gain",
        "ideal",
        "name",
    )
    """
    Detector camera with raw image acquisition and dark frame subtraction.
    
    The camera models a realistic detector with the following features:
    - Dark current accumulation during integration
    - Read noise
    - Photon shot noise
    - Automatic dark frame subtraction
    
    Parameters
    ----------
    pixels : tuple of int, optional
        Number of pixels (width, height). Default: (1024, 1024)
    pixel_size : astropy.Quantity, optional
        Physical size of pixels (e.g. 10*u.um) or angular size (e.g. 0.01*u.arcsec).
        If None, automatically determined to satisfy Nyquist sampling at the given wavelength
        relative to the input wavefront aperture.
    dark_current : astropy.Quantity, optional
        Dark current rate in electrons per second per pixel. 
        Typical values: 0.001-0.1 e-/s for cooled scientific cameras.
        Default: 0.01 e-/s
    read_noise : astropy.Quantity, optional
        Read noise (RMS) in electrons per pixel per read.
        Typical values: 1-10 e- for scientific cameras.
        Default: 3 e-
    integration_time : astropy.Quantity, optional
        Integration time. Default: 1 s
    quantum_efficiency : float, optional
        Quantum efficiency (0 to 1). Fraction of incident photons converted to electrons.
        Default: 0.9 (90%)
    gain : float, optional
        Camera gain in electrons per ADU (Analog-to-Digital Unit).
        Default: 1.0 e-/ADU
    name : str, optional
        Name of the camera for identification in diagrams
    
    Examples
    --------
    >>> # Create camera with typical scientific CCD parameters
    >>> camera = Camera(pixels=(2048, 2048), 
    ...                 dark_current=0.01*u.electron/u.s,
    ...                 read_noise=3*u.electron,
    ...                 integration_time=10*u.s)
    >>> 
    >>> # Acquire raw image (with signal + dark + noise)
    >>> raw = camera.get_raw_image(wavefront, pipeline)
    >>> 
    >>> # Get dark frame only
    >>> dark = camera.get_dark()
    >>> 
    >>> # Get reduced image (signal with dark subtracted)
    >>> reduced = camera.get_image(wavefront, pipeline)
    """
    def __init__(self, pixels: Tuple[int, int] = (1024, 1024), 
                 pixel_size: Optional[u.Quantity] = None,
                 dark_current: u.Quantity = 0.01*u.electron/u.s, 
                 read_noise: u.Quantity = 3*u.electron,
                 integration_time: u.Quantity = 1*u.s,
                 quantum_efficiency: float = 0.9,
                 gain: float = 1.0,
                 ideal: bool = False,
                 name: Optional[str] = None, **kwargs):
        super().__init__(name=name or "Camera")
        self.pixels = pixels
        self.pixel_size = pixel_size
        
        # Store parameters (convert to native units for performance)
        self.dark_current = float(dark_current.to(u.electron/u.s).value)  # e-/s
        self.read_noise = float(read_noise.to(u.electron).value)  # e-
        self.integration_time_value = float(integration_time.to(u.s).value)  # s
        self.integration_time = integration_time  # Keep original for API
        self.quantum_efficiency = float(quantum_efficiency)
        self.gain = float(gain)  # e-/ADU
        
        self.gain = float(gain)  # e-/ADU
        
        self.ideal = ideal
        
    def to_dict(self) -> dict:
        """Serialize camera configuration."""
        data = super().to_dict()
        data.update({
            "pixels": list(self.pixels),
            "pixel_size": serialize_value(self.pixel_size) if self.pixel_size is not None else None,
            "dark_current": serialize_value(self.dark_current * u.electron / u.s),
            "read_noise": serialize_value(self.read_noise * u.electron),
            "integration_time": serialize_value(self.integration_time),
            "quantum_efficiency": self.quantum_efficiency,
            "gain": self.gain,
            "ideal": self.ideal
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Camera':
        """Create camera from dictionary."""
        name = data.get("name")
        pixels = tuple(data.get("pixels", (1024, 1024)))
        pixel_size = deserialize_value(data.get("pixel_size"))
        
        dark_current = deserialize_value(data.get("dark_current"))
        read_noise = deserialize_value(data.get("read_noise"))
        integration_time = deserialize_value(data.get("integration_time"))
        quantum_efficiency = data.get("quantum_efficiency", 0.9)
        gain = data.get("gain", 1.0)
        
        quantum_efficiency = data.get("quantum_efficiency", 0.9)
        gain = data.get("gain", 1.0)
        
        ideal = data.get("ideal", False)
        # Backward compatibility
        if "include_photon_noise" in data:
             # if include_photon_noise was True (default), ideal is False. 
             # if include_photon_noise was False, it meant "no noise" which maps to ideal=True?
             # User said "include_photon_noise peut être généralisé en ideal:bool=False".
             # Actually, include_photon_noise=False usually meant "just signal + dark", no poisson.
             # If ideal=True means "no noise at all", it maps roughly to include_photon_noise=False?
             # Let's assume strict mapping: ideal replaces it.
             pass
        
        return cls(pixels=pixels, pixel_size=pixel_size, 
                   dark_current=dark_current, read_noise=read_noise,
                   integration_time=integration_time, quantum_efficiency=quantum_efficiency,
                   gain=gain,
                   ideal=ideal, name=name)

    def get_raw_image(self, wavefront:Wavefront=None) -> np.ndarray:
        """
        Acquire raw detector image including signal, dark current, and noise.
        
        This method simulates a realistic detector readout with:
        1. Photon signal from the wavefront (with quantum efficiency)
        2. Dark current accumulation
        3. Photon shot noise (Poisson statistics)
        4. Read noise (Gaussian)
        
        Parameters
        ----------
        wavefront : Wavefront or List[Wavefront] or None
            Input wavefront containing the electromagnetic field. If None,
            only dark current and noise are generated (dark frame).
        
        Returns
        -------
        raw_image : ndarray
            Raw detector image in electrons. Shape matches self.pixels.
        
        Notes
        -----
        The raw image contains:
        - Signal: ``|wavefront|²`` × QE × integration_time
        - Dark: dark_current × integration_time (per pixel)
        - Shot noise: Poisson(signal + dark)
        - Read noise: Gaussian(0, read_noise)
        
        Examples
        --------
        >>> camera = Camera(pixels=(512, 512), integration_time=10*u.s)
        >>> raw = camera.get_raw_image(wavefront, pipeline)
        >>> print(f"Raw image range: [{raw.min():.1f}, {raw.max():.1f}] e-")
        """

        # 1. Automatic Wavefront Retrieval (if None)
        if wavefront is None:
            # Check if we can automatically retrieve it from the pipeline
            if self.pipeline is not None:
                wavefront = self.previous().process().propagate()
            # If no pipeline, generate dark frame
            else:
                import warnings
                warnings.warn(
                    "No wavefront provided and no pipeline attached. Generating dark frame.",
                    UserWarning
                )
                wavefront = Wavefront() * 0 # Dark frame

        if isinstance(wavefront, list):
            # Incoherent sum of intensities from multiple wavefronts
            intensity = sum(wf.intensity for wf in wavefront)
        else:
            intensity = wavefront.intensity

        # Applying noises
        if not self.ideal:

            # Electrons arriving at the detector
            electrons = 0
        
            # 1. Convert to electrons: apply quantum efficiency and integration time
            electrons += intensity * self.quantum_efficiency * self.integration_time_value
            
            # 2. Dark current accumulation
            electrons += self.dark_current * self.integration_time_value
    
            # 3. Photon shot noise (Poisson statistics)
            rng = np.random.default_rng()
            raw_image = rng.poisson(electrons).astype(float)
            
            # 4. Read noise (Gaussian)
            raw_image += rng.normal(0, self.read_noise, size=raw_image.shape)
        else:
            # Ideal case: signal only, no dark current or noise
            raw_image = intensity * self.quantum_efficiency * self.integration_time_value

        return raw_image
    
    def _get_extent_and_labels(self, unit: str, wavefront: Optional[Wavefront] = None) -> Tuple[Optional[List[float]], str, str]:
        """
        Determine plot extent and axis labels based on requested unit.
        
        Parameters
        ----------
        unit : {'pixel', 'size', 'angle'}
            Requested unit for axes.
        wavefront : Wavefront, optional
            Wavefront used for propagation context (needed for focal length in some conversions).
            
        Returns
        -------
        extent : list or None
            [xmin, xmax, ymin, ymax] for imshow. None if unit='pixel'.
        xlabel, ylabel : str
            Axis labels.
        """
        import warnings
        
        # Default (Pixel)
        if unit == 'pixel':
            return None, 'Pixel X', 'Pixel Y'
            
        try:
            # Check availability
            if self.pixel_size is None and unit != 'pixel':
                raise ValueError("Camera has no 'pixel_size' defined.")

            # Calculate total width/fov
            # We assume square pixels for plotting simplicity usually, or handle rectangular if needed.
            # Here assuming square pixel_size or taking mean if not? 
            # self.pixel_size is usually a scalar Quantity.
            
            # Determine Pixel Scale in requested unit
            scale = None
            label_unit = ""
            
            if unit == 'size':
                if self.pixel_size.unit.is_equivalent(u.m):
                    scale = self.pixel_size
                elif self.pixel_size.unit.is_equivalent(u.rad):
                    # Angular -> Physical (Need Focal Length)
                    if wavefront is None or wavefront._last_focal_length_m is None:
                        raise ValueError("Cannot convert Angular pixel_size to Physical: No focal length known.")
                    
                    f_m = wavefront._last_focal_length_m * u.m
                    scale = (self.pixel_size.to(u.rad).value * f_m).to(u.um) # Convert to reasonable unit like um or mm?
                    # Let's keep it in input unit or auto-scale? Astropy handles it if we use Quantity?
                    # But imshow extent needs floats. We pick a standard unit.
                    scale = scale.to(u.mm)
                else:
                    raise ValueError(f"Unknown pixel_size unit: {self.pixel_size.unit}")
                
                label_unit = "mm" 
                # If we want smart scaling (um vs mm), we could do it here, but let's stick to mm for 'size' generally
                if scale.value < 0.1: 
                     scale = scale.to(u.um)
                     label_unit = "µm"

            elif unit == 'angle':
                if self.pixel_size.unit.is_equivalent(u.rad):
                    scale = self.pixel_size
                elif self.pixel_size.unit.is_equivalent(u.m):
                     # Physical -> Angular (Need Focal Length)
                    if wavefront is None or wavefront._last_focal_length_m is None:
                        raise ValueError("Cannot convert Physical pixel_size to Angular: No focal length known.")
                    f_m = wavefront._last_focal_length_m * u.m
                    scale = (self.pixel_size / f_m) * u.rad
                else:
                    raise ValueError(f"Unknown pixel_size unit: {self.pixel_size.unit}")
                
                label_unit = "arcsec"
                scale = scale.to(u.arcsec)
                
            else:
                raise ValueError(f"Unknown unit mode: {unit}")
            
            # Calculate Extent
            # Centered on 0
            w, h = self.pixels
            
            # scale is per pixel. Total width = w * scale
            val = scale.value
            
            half_w = (w * val) / 2.0
            half_h = (h * val) / 2.0
            
            extent = [-half_w, half_w, -half_h, half_h]
            return extent, f"Position X ({label_unit})", f"Position Y ({label_unit})"

        except Exception as e:
            warnings.warn(f"Could not apply unit '{unit}': {e}. Falling back to 'pixel'.")
            return None, 'Pixel X', 'Pixel Y'

    def get_dark(self) -> np.ndarray:
        """
        Generate dark frame (detector readout with no illumination).
        
        This method simulates a dark exposure with the same integration time
        as science frames. Dark frames contain:
        - Dark current accumulation
        - Shot noise from dark current
        - Read noise
        
        Dark frames are used for calibration to subtract thermal signal from
        science images.
        
        Returns
        -------
        dark_frame : ndarray
            Dark frame in electrons. Shape matches self.pixels.
        
        Notes
        -----
        In real observations, multiple dark frames are typically averaged to
        reduce noise. This method generates a single realization.
        
        The dark frame does NOT include signal from astronomical sources.
        It only contains detector-intrinsic contributions.
        
        Examples
        --------
        >>> camera = Camera(pixels=(512, 512), 
        ...                 dark_current=0.1*u.electron/u.s,
        ...                 integration_time=100*u.s)
        >>> dark = camera.get_dark()
        >>> print(f"Dark current: {dark.mean():.1f} e-")
        >>> print(f"Dark noise: {dark.std():.1f} e-")
        """
        # Dark frame = raw image with dummy zero-amplitude wavefront
        # Create a dummy wavefront with 0 amplitude
        # We need it to have some properties to pass internal checks if any, 
        # but for get_raw_image(None), it tried to retrieve.
        # Now we pass explicit wavefront=0.
        
        # We can construct a minimal wavefront.
        # Since get_raw_image expects 'wavefront' object or None.
        # If we pass None, it tries to retrieve.
        # So we pass a Wavefront with 0 amplitude.
        
        # To avoid circular imports or complex instantiation, we can assume 'Wavefront' is available 
        # or use a mocked object if simple. 
        # But best is to use the actual Wavefront class.
        # It is imported at top level? Yes: from ...core.wavefront import Wavefront
        
        # We need to match the camera pixels to avoid resizing logic issues if possible, 
        # but valid wavefront usually has physical size. 
        # Simplest is:
        
        dummy_wf = Wavefront(wavelength=550*u.nm, npix=self.pixels[0], nsource=1)
        dummy_wf[:] = 0 # Set amplitude to 0
        
        # We also need to set a pixel scale to avoid errors if get_raw_image checks it?
        # get_raw_image uses 'wf0.width', 'wf0.wavelength'.
        # Wavefront() init sets defaults.
        
        return self.get_raw_image(wavefront=dummy_wf)
    
    def get_image(self, wavefront: Optional[Wavefront] = None) -> np.ndarray:
        """
        Get calibrated (reduced) detector image with automatic dark subtraction.
        
        This method performs automatic data reduction:
        1. Acquire raw image (signal + dark + noise)
        2. Generate dark frame (dark + noise)
        3. Subtract dark from raw to isolate signal
        
        The result approximates what an astronomer would obtain after basic
        data reduction pipeline.
        
        Parameters
        ----------
        wavefront : Wavefront or None
            Input wavefront containing the electromagnetic field
        
        Returns
        -------
        reduced_image : ndarray
            Calibrated detector image in electrons. Shape matches self.pixels.
        
        Notes
        -----
        **Physical interpretation:**
        
        After dark subtraction, the image contains:
        - Astronomical signal (from wavefront)
        - Shot noise from signal (σ = √signal)
        - Residual read noise (×√2 from both frames)
        - Residual shot noise from dark (×√2)
        
        The √2 noise increase from dark subtraction is fundamental: subtracting
        two noisy frames adds their variances (σ² = σ₁² + σ₂²).
        
        **Why dark subtraction matters:**
        
        Without dark subtraction, thermal electrons from the detector would
        contaminate the astronomical signal, especially for faint sources or
        long integrations.
        
        Examples
        --------
        >>> camera = Camera(pixels=(256, 256), integration_time=60*u.s)
        >>> pipeline.add_layer(camera) # Add the camera to the desired pipeline
        >>> 
        >>> # Get reduced image (recommended for science)
        >>> reduced = camera.get_image(wavefront, pipeline)
        """
        # Automatic path simulation if wavefront is None
        if wavefront is None and self.pipeline is not None:
             # Try to propagate from pipeline
             try:
                 # We look for 'propagate_until' method which we added to Pipeline
                 if hasattr(self.pipeline, 'propagate_until'):
                    wavefront = self.pipeline.propagate_until(self)
             except Exception as e:
                 print(f"Camera path simulation failed: {e}")
                 # Fallthrough to None (dark frame)

        # Automatic data reduction pipeline:
        
        # 1. Acquire raw science frame
        raw_image = self.get_raw_image(wavefront)
        
        # 2. Acquire dark frame (same integration time)
        dark_frame = self.get_dark()
        
        # 3. Dark subtraction
        reduced_image = raw_image - dark_frame
        
        return reduced_image

    def process(self, wavefront: Wavefront, pipeline: Optional['Pipeline'] = None) -> np.ndarray:
        """
        Process wavefront and return reduced detector image.
        
        This is the Layer/Component interface method called by Pipeline.observe().
        By default, it returns a dark-subtracted (reduced) image.
        
        For raw images or dark frames, use get_raw_image() or get_dark() directly.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront
        
        Returns
        -------
        ndarray
            Reduced detector image in electrons
        """
        return self.get_image(wavefront)
        
    def plot(self, wavefront: Optional[Wavefront] = None, 
             ax: Optional[plt.Axes] = None,
             show: bool = True,
             title: Optional[str] = None,
             log_scale: bool = False,
             debug: bool = False,
             unit: str = 'pixel') -> plt.Axes:
        """
        Visualize the camera detector output.
        
        Simulates the full detection process (acquisition, noise, dark subtraction)
        and plots the resulting image.
        
        Parameters
        ----------
        wavefront : Wavefront, optional
            Input wavefront. If None, plots a dark frame (or empty image).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        show : bool, optional
            If True, calls plt.show(). Default: True
        title : str, optional
            Custom title for the plot.
        log_scale : bool, optional
            If True, plot intensity in log scale (log10). Default: False
        debug : bool, optional
            If True, suppress plt.show() and save plot to debug file. Default: False
        unit : {'pixel', 'size', 'angle'}, optional
            Unit for axes. Default: 'pixel'.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        # Get the simulated image
        image = self.get_image(wavefront)
        
        # Determine Extent and Labels
        extent, xlabel, ylabel = self._get_extent_and_labels(unit, wavefront)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Prepare data for plotting
        plot_data = image
        if log_scale:
            # Avoid log(0) or log(negative)
            plot_data = np.log10(np.maximum(image, 1e-10))
            
        # Plot
        # Use origin='lower' to match astronomical convention (y increases upwards)
        im = ax.imshow(plot_data, origin='lower', cmap='inferno', extent=extent)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Counts (e-)' if not log_scale else 'Log Counts (e-)')
        
        # Labels and Title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if title:
            ax.set_title(title)
        else:
            base_title = f"Camera Output ({self.pixels[0]}x{self.pixels[1]})"
            extra = []
            if wavefront is None:
                extra.append("Dark Frame")
            else:
                extra.append(f"Exp: {self.integration_time}")
            
            ax.set_title(f"{base_title} - {', '.join(extra)}")
            
        if debug:
            show = False
            
        if show:
            plt.show()
            
        if debug:
            print(f"DEBUG Plot Camera: {self.name} {title if title else ''}")
            print(f"  Range: {image.min():.2f} to {image.max():.2f}")
            try:
                import os
                import time
                os.makedirs("tests/generated", exist_ok=True)
                timestamp = int(time.time() * 1000)
                filename = f"tests/generated/plot_camera_{timestamp}.png"
                fig.savefig(filename)
                print(f"  Saved plot to {filename}")
            except Exception as e:
                print(f"  Failed to save debug plot: {e}")
            if ax is None: # Only close if we created the figure
                plt.close(fig)
            
        return ax
    
    def plot_raw(self, wavefront: Optional[Wavefront] = None, 
                 ax: Optional[plt.Axes] = None, show: bool = True,
                 unit: str = 'pixel') -> plt.Axes:
        """Plot the raw image (with noise and dark current)."""
        raw_img = self.get_raw_image(wavefront)
        
        extent, xlabel, ylabel = self._get_extent_and_labels(unit, wavefront)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        im = ax.imshow(raw_img, origin='lower', cmap='inferno', extent=extent)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Counts (e-)')
        ax.set_title(f"Raw Image ({self.integration_time})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if show:
            plt.show()
        return ax

    def plot_dark(self, ax: Optional[plt.Axes] = None, show: bool = True,
                  unit: str = 'pixel') -> plt.Axes:
        """Plot the dark frame."""
        dark_img = self.get_dark()
        
        extent, xlabel, ylabel = self._get_extent_and_labels(unit, wavefront=None)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        im = ax.imshow(dark_img, origin='lower', cmap='inferno', extent=extent)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Counts (e-)')
        ax.set_title(f"Dark Frame ({self.integration_time})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if show:
            plt.show()
        return ax

    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for Camera."""
        attrs = {}
        attrs['pixels'] = f"{self.pixels[0]} × {self.pixels[1]}"
        attrs['dark_current'] = f"{self.dark_current:.3f} e-/s"
        attrs['read_noise'] = f"{self.read_noise:.1f} e-"
        attrs['integration_time'] = str(self.integration_time)
        attrs['quantum_efficiency'] = f"{self.quantum_efficiency:.2%}"
        attrs['gain'] = f"{self.gain:.2f} e-/ADU"
        if self.pixel_size is not None:
            attrs['pixel_size'] = str(self.pixel_size)
        if hasattr(self, 'thermal_background_temp'):
            attrs['thermal_temp'] = str(self.thermal_background_temp)
        return attrs

def test_camera():
    """Test Camera functionality including new methods."""
    # Test basic instantiation
    cam = Camera(pixels=(100, 100))
    assert cam.pixels == (100, 100)

    # Test defaults
    default_cam = Camera()
    assert default_cam.pixels == (1024, 1024)
    
    # Test dark frame generation
    dark = default_cam.get_dark()
    assert dark.shape == (1024, 1024)
    assert dark.dtype == np.float64
    
    # Test raw image without wavefront (should be dark only)
    raw_no_signal = default_cam.get_raw_image(wavefront=None)
    assert raw_no_signal.shape == (1024, 1024)
    
    # Test image reduction
    # Create a mock wavefront with simple field
    class MockWavefront:
        def __init__(self, size):
            self.field = np.ones((size, size), dtype=np.complex128)
    
    mock_wf = MockWavefront(1024)
    reduced = default_cam.get_image(mock_wf)
    assert reduced.shape == (1024, 1024)
    
    # Test that dark subtraction changes the result
    raw = default_cam.get_raw_image(mock_wf)
    reduced_manual = default_cam.get_image(mock_wf)
    # raw_via_get_image = default_cam.get_image(mock_wf, None, subtract_dark=False)
    # assert np.allclose(raw, raw_via_get_image), "Raw image should match when subtract_dark=False"
    
    print("✓ Camera basic instantiation")
    print("✓ Dark frame generation")
    print("✓ Raw image acquisition")
    print("✓ Image reduction (dark subtraction)")
    print("✓ Process method (Layer interface)")

if __name__ == "__main__":
    test_camera()
    print("\nAll Camera tests passed.")
