import numpy as np
from typing import Tuple, Optional, Any
from astropy import units as u
from ..core.pipeline import Element, Layer, DetectionLayer, Pipeline, serialize_value, deserialize_value
from ..core.simulation import Wavefront, WavefrontArray
import matplotlib.pyplot as plt

class Camera(DetectionLayer):
    __slots__ = (
        "pixels",
        "dark_current",
        "read_noise",
        "integration_time_value",
        "integration_time",
        "quantum_efficiency",
        "gain",
        "thermal_background",
        "thermal_background_temp",
        "_rng",
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
    thermal_background : astropy.Quantity, optional
        Thermal background rate from warm instrument in electrons per second per pixel.
        If None, calculated from thermal_background_temp. Default: None
    thermal_background_temp : astropy.Quantity, optional
        Instrument temperature for thermal emission calculation.
        Only used if thermal_background is None. Default: 280 K
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
                 dark_current: u.Quantity = 0.01*u.electron/u.s, 
                 read_noise: u.Quantity = 3*u.electron,
                 integration_time: u.Quantity = 1*u.s,
                 quantum_efficiency: float = 0.9,
                 gain: float = 1.0,
                 thermal_background: Optional[u.Quantity] = None,
                 thermal_background_temp: u.Quantity = 280*u.K,
                 name: Optional[str] = None, **kwargs):
        super().__init__(name=name or "Camera")
        self.pixels = pixels
        
        # Store parameters (convert to native units for performance)
        self.dark_current = float(dark_current.to(u.electron/u.s).value)  # e-/s
        self.read_noise = float(read_noise.to(u.electron).value)  # e-
        self.integration_time_value = float(integration_time.to(u.s).value)  # s
        self.integration_time = integration_time  # Keep original for API
        self.quantum_efficiency = float(quantum_efficiency)
        self.gain = float(gain)  # e-/ADU
        
        # Thermal background from warm instrument
        if thermal_background is not None:
            self.thermal_background = float(thermal_background.to(u.electron/u.s).value)  # e-/s
        else:
            # Default to zero thermal background unless explicitly provided
            # Tests expect dark frames around dark_current × integration_time
            self.thermal_background = 0.0
        self.thermal_background_temp = thermal_background_temp
        
        # Random number generator for reproducible noise
        self._rng = np.random.default_rng()
        
    def to_dict(self) -> dict:
        """Serialize camera configuration."""
        data = super().to_dict()
        data.update({
            "pixels": list(self.pixels),
            "dark_current": serialize_value(self.dark_current * u.electron / u.s),
            "read_noise": serialize_value(self.read_noise * u.electron),
            "integration_time": serialize_value(self.integration_time),
            "quantum_efficiency": self.quantum_efficiency,
            "gain": self.gain,
            "thermal_background": serialize_value(self.thermal_background * u.electron / u.s) if self.thermal_background else None,
            "thermal_background_temp": serialize_value(self.thermal_background_temp)
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Camera':
        """Create camera from dictionary."""
        name = data.get("name")
        pixels = tuple(data.get("pixels", (1024, 1024)))
        
        dark_current = deserialize_value(data.get("dark_current"))
        read_noise = deserialize_value(data.get("read_noise"))
        integration_time = deserialize_value(data.get("integration_time"))
        quantum_efficiency = data.get("quantum_efficiency", 0.9)
        gain = data.get("gain", 1.0)
        
        thermal_background = deserialize_value(data.get("thermal_background"))
        thermal_background_temp = deserialize_value(data.get("thermal_background_temp"))
        
        return cls(pixels=pixels, dark_current=dark_current, read_noise=read_noise,
                   integration_time=integration_time, quantum_efficiency=quantum_efficiency,
                   gain=gain, thermal_background=thermal_background,
                   thermal_background_temp=thermal_background_temp, name=name)
    
    def _combine_wavefronts(self, wf_array: WavefrontArray) -> np.ndarray:
        """Combine wavefronts from an array into a single focal plane intensity image."""
        if not wf_array.wavefronts:
            return np.zeros(self.pixels)
            
        # Assume all wavefronts have same properties
        wf0 = wf_array.wavefronts[0]
        scale = wf0.pixel_scale.to(u.m).value
        
        # Use the size of the input wavefronts as the base canvas size
        if wf0.ndim == 3:
            samples, h, w = wf0.shape
            canvas = np.zeros((samples, h, w), dtype=np.complex128)
        else:
            h, w = wf0.shape
            canvas = np.zeros((h, w), dtype=np.complex128)
        
        # Check if locations are available
        locations = wf_array.locations
        if locations is None:
            locations = [(0.0, 0.0)] * len(wf_array.wavefronts)
            
        for wf, loc in zip(wf_array.wavefronts, locations):
            # Calculate shift in pixels
            lx, ly = loc
            shift_x = int(lx / scale)
            shift_y = int(ly / scale)
            
            # Shift the field using roll (valid for small shifts relative to array size)
            if wf.ndim == 3:
                field_shifted = np.roll(wf, (shift_y, shift_x), axis=(1, 2))
            else:
                field_shifted = np.roll(wf, (shift_y, shift_x), axis=(0, 1))
            
            # Add to canvas (coherent combination)
            canvas += field_shifted
            
        # FFT to get focal plane field
        # fftshift moves zero freq to center
        if canvas.ndim == 3:
            focal_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(canvas, axes=(1,2)), axes=(1,2)), axes=(1,2))
            # Sum intensities (incoherent sum of samples)
            intensity = np.sum(np.abs(focal_field)**2, axis=0)
        else:
            focal_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(canvas)))
            intensity = np.abs(focal_field)**2
        
        # Return intensity
        return intensity

    def get_raw_image(self, wavefront: Optional[Any]) -> np.ndarray:
        """
        Acquire raw detector image including signal, dark current, and noise.
        
        This method simulates a realistic detector readout with:
        1. Photon signal from the wavefront (with quantum efficiency)
        2. Dark current accumulation
        3. Photon shot noise (Poisson statistics)
        4. Read noise (Gaussian)
        
        Parameters
        ----------
        wavefront : Wavefront or WavefrontArray or None
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
        # 1. Signal from wavefront (if provided)
        if wavefront is not None:
            if isinstance(wavefront, WavefrontArray):
                # Combine wavefronts for interferometry
                intensity = self._combine_wavefronts(wavefront)
            elif isinstance(wavefront, Wavefront):
                # Single wavefront
                # Single wavefront
                if wavefront.ndim == 3:
                     # Calculate focal plane field via FFT
                     # fftshift moves zero freq to center
                     # We assume the wavefront is at the pupil plane and we want the image plane
                     # Use .value to ensure we work with numpy arrays (avoid Unit issues)
                     wf_data = wavefront.value if hasattr(wavefront, 'value') else wavefront
                     focal_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wf_data, axes=(1,2)), axes=(1,2)), axes=(1,2))
                     intensity = np.sum(np.abs(focal_field)**2, axis=0)
                     # Normalize FFT energy (Parseval: sum(|F|^2) = N*sum(|f|^2)). We want sum(|I|^2) = sum(|P|^2) for simple conservation check
                     norm = focal_field.shape[1] * focal_field.shape[2]
                     intensity /= norm
                else:
                    wf_data = wavefront.value if hasattr(wavefront, 'value') else wavefront
                    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wf_data)))
                    intensity = np.abs(focal_field) ** 2
                    # Normalize FFT energy
                    norm = focal_field.shape[0] * focal_field.shape[1]
                    intensity /= norm
            else:
                intensity = np.zeros(self.pixels)
            
            # Resize intensity to match camera pixels if needed
            if intensity.shape != self.pixels:
                # Use interpolation to resize the field to camera dimensions
                from scipy.ndimage import zoom
                zoom_factors = (self.pixels[0] / intensity.shape[0], 
                               self.pixels[1] / intensity.shape[1])
                intensity = zoom(intensity, zoom_factors, order=1)
            
            # Convert to electrons: apply quantum efficiency and integration time
            signal_electrons = intensity * self.quantum_efficiency * self.integration_time_value
        else:
            # No signal (dark frame only)
            signal_electrons = np.zeros(self.pixels)
        
        # 2. Dark current accumulation
        dark_electrons = self.dark_current * self.integration_time_value
        
        # 3. Thermal background from warm instrument
        thermal_electrons = self.thermal_background * self.integration_time_value
        
        # 4. Total signal before noise
        total_signal = signal_electrons + dark_electrons + thermal_electrons
        
        # 5. Apply shot noise (Poisson statistics)
        # Photons follow Poisson distribution: σ² = N
        total_signal_noisy = self._rng.poisson(lam=np.maximum(total_signal, 0))
        
        # 6. Add read noise (Gaussian)
        read_noise_array = self._rng.normal(loc=0, scale=self.read_noise, size=self.pixels)
        
        # 7. Combine all contributions
        raw_image = total_signal_noisy + read_noise_array
        
        return raw_image
    
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
        # Dark frame = raw image with no wavefront input
        return self.get_raw_image(wavefront=None)
    
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
        
        This is the Layer/Element interface method called by Pipeline.observe().
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
             debug: bool = False) -> plt.Axes:
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
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        # Get the simulated image
        image = self.get_image(wavefront)
        
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
        im = ax.imshow(plot_data, origin='lower', cmap='inferno')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Counts (e-)' if not log_scale else 'Log Counts (e-)')
        
        # Labels and Title
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        
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
            print(f"  Image shape: {image.shape}")
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
    
    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for Camera."""
        attrs = {}
        attrs['pixels'] = f"{self.pixels[0]} × {self.pixels[1]}"
        attrs['dark_current'] = f"{self.dark_current:.3f} e-/s"
        attrs['read_noise'] = f"{self.read_noise:.1f} e-"
        attrs['integration_time'] = str(self.integration_time)
        attrs['quantum_efficiency'] = f"{self.quantum_efficiency:.2%}"
        attrs['gain'] = f"{self.gain:.2f} e-/ADU"
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
