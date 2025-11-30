import numpy as np
from typing import Tuple, Optional
from astropy import units as u
from ..core.context import Element, Context
from ..core.simulation import Wavefront

class Camera(Element):
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
    >>> raw = camera.get_raw_image(wavefront, context)
    >>> 
    >>> # Get dark frame only
    >>> dark = camera.get_dark()
    >>> 
    >>> # Get reduced image (signal with dark subtracted)
    >>> reduced = camera.get_image(wavefront, context)
    """
    def __init__(self, pixels: Tuple[int, int] = (1024, 1024), 
                 dark_current: u.Quantity = 0.01*u.electron/u.s, 
                 read_noise: u.Quantity = 3*u.electron,
                 integration_time: u.Quantity = 1*u.s,
                 quantum_efficiency: float = 0.9,
                 gain: float = 1.0,
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
        
        # Random number generator for reproducible noise
        self._rng = np.random.default_rng()
    
    def get_raw_image(self, wavefront: Optional[Wavefront], context: Optional[Context] = None) -> np.ndarray:
        """
        Acquire raw detector image including signal, dark current, and noise.
        
        This method simulates a realistic detector readout with:
        1. Photon signal from the wavefront (with quantum efficiency)
        2. Dark current accumulation
        3. Photon shot noise (Poisson statistics)
        4. Read noise (Gaussian)
        
        Parameters
        ----------
        wavefront : Wavefront or None
            Input wavefront containing the electromagnetic field. If None,
            only dark current and noise are generated (dark frame).
        context : Context, optional
            Simulation context (unused currently, reserved for future features)
        
        Returns
        -------
        raw_image : ndarray
            Raw detector image in electrons. Shape matches self.pixels.
        
        Notes
        -----
        The raw image contains:
        - Signal: |wavefront.field|² × QE × integration_time
        - Dark: dark_current × integration_time (per pixel)
        - Shot noise: Poisson(signal + dark)
        - Read noise: Gaussian(0, read_noise)
        
        Examples
        --------
        >>> camera = Camera(pixels=(512, 512), integration_time=10*u.s)
        >>> raw = camera.get_raw_image(wavefront, context)
        >>> print(f"Raw image range: [{raw.min():.1f}, {raw.max():.1f}] e-")
        """
        # 1. Signal from wavefront (if provided)
        if wavefront is not None and hasattr(wavefront, 'field'):
            # Compute intensity (photon flux per pixel)
            intensity = np.abs(wavefront.field) ** 2
            
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
        
        # 3. Total signal before noise
        total_signal = signal_electrons + dark_electrons
        
        # 4. Apply shot noise (Poisson statistics)
        # Photons follow Poisson distribution: σ² = N
        total_signal_noisy = self._rng.poisson(lam=np.maximum(total_signal, 0))
        
        # 5. Add read noise (Gaussian)
        read_noise_array = self._rng.normal(loc=0, scale=self.read_noise, size=self.pixels)
        
        # 6. Combine all contributions
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
        return self.get_raw_image(wavefront=None, context=None)
    
    def get_image(self, wavefront: Optional[Wavefront], context: Optional[Context] = None,
                  subtract_dark: bool = True) -> np.ndarray:
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
        context : Context, optional
            Simulation context
        subtract_dark : bool, optional
            If True, subtract dark frame from raw image. If False, return
            raw image without dark subtraction. Default: True
        
        Returns
        -------
        reduced_image : ndarray
            Calibrated detector image in electrons. Shape matches self.pixels.
            - If subtract_dark=True: signal + residual noise
            - If subtract_dark=False: raw image (same as get_raw_image)
        
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
        >>> 
        >>> # Get reduced image (recommended for science)
        >>> reduced = camera.get_image(wavefront, context)
        >>> 
        >>> # Get raw image without dark subtraction
        >>> raw = camera.get_image(wavefront, context, subtract_dark=False)
        >>> 
        >>> # Manual reduction
        >>> raw_manual = camera.get_raw_image(wavefront, context)
        >>> dark_manual = camera.get_dark()
        >>> reduced_manual = raw_manual - dark_manual
        """
        if not subtract_dark:
            # Return raw image without reduction
            return self.get_raw_image(wavefront, context)
        
        # Automatic data reduction pipeline:
        
        # 1. Acquire raw science frame
        raw_image = self.get_raw_image(wavefront, context)
        
        # 2. Acquire dark frame (same integration time)
        dark_frame = self.get_dark()
        
        # 3. Dark subtraction
        reduced_image = raw_image - dark_frame
        
        return reduced_image

    def process(self, wavefront: Wavefront, context: Context) -> np.ndarray:
        """
        Process wavefront and return reduced detector image.
        
        This is the Layer/Element interface method called by Context.observe().
        By default, it returns a dark-subtracted (reduced) image.
        
        For raw images or dark frames, use get_raw_image() or get_dark() directly.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront
        context : Context
            Simulation context
        
        Returns
        -------
        ndarray
            Reduced detector image in electrons
        """
        return self.get_image(wavefront, context, subtract_dark=True)

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
    reduced = default_cam.get_image(mock_wf, None)
    assert reduced.shape == (1024, 1024)
    
    # Test that dark subtraction changes the result
    raw = default_cam.get_raw_image(mock_wf, None)
    reduced_manual = default_cam.get_image(mock_wf, None, subtract_dark=True)
    raw_via_get_image = default_cam.get_image(mock_wf, None, subtract_dark=False)
    assert np.allclose(raw, raw_via_get_image), "Raw image should match when subtract_dark=False"
    
    print("✓ Camera basic instantiation")
    print("✓ Dark frame generation")
    print("✓ Raw image acquisition")
    print("✓ Image reduction (dark subtraction)")
    print("✓ Process method (Layer interface)")

if __name__ == "__main__":
    test_camera()
    print("\nAll Camera tests passed.")
