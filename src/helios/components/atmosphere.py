"""Atmospheric turbulence and adaptive optics correction.

This module provides atmospheric turbulence modeling with frozen-flow temporal evolution
and Zernike-based adaptive optics correction.
"""
import numpy as np
import math
from astropy import units as u
from typing import Tuple, List, Union, Optional
import matplotlib.pyplot as _plt

from ..core.context import Element, Context
from ..core.simulation import Wavefront, WavefrontArray
from .collector import TelescopeArray


class Atmosphere(Element):
    """Kolmogorov atmosphere layer producing chromatic phase screens with frozen-flow turbulence.

    The atmosphere introduces optical path difference (OPD) errors that are chromatic - 
    the phase shift depends on wavelength: phi = 2π * OPD / λ.

    Temporal evolution is modeled via frozen-flow (Taylor hypothesis): turbulent screens 
    drift at constant wind velocity, and different observation times sample different 
    regions of the frozen turbulent volume.

    Parameters
    ----------
    rms : astropy.Quantity
        Desired RMS of the OPD (optical path difference) in length units (e.g., nm, μm).
        This is the wavefront error amplitude, NOT phase in radians.
        Default: 100 nm (good seeing conditions at visible wavelengths).
    
    wind_speed : astropy.Quantity
        Wind velocity vector magnitude and direction. Can be:
        - Scalar Quantity: wind speed in m/s (default direction: +x)
        - Tuple of 2 Quantities: (vx, vy) wind velocity components in m/s
        Default: 5 m/s in +x direction (~18 km/h, typical high-altitude wind).
    
    wind_direction : float, optional
        Wind direction in degrees (0° = +x, 90° = +y). Used only if wind_speed is scalar.
        Default: 0°.
    
    seed : int, optional
        RNG seed for reproducible turbulent realizations. If None, uses random seed.
        The same seed produces the same frozen turbulent volume.
        Default: None (random).
    
    inner_scale : astropy.Quantity, optional
        Inner scale of turbulence (l0) in meters. Below this scale, turbulence becomes 
        isotropic. Default: None (pure Kolmogorov, no inner scale).
    
    outer_scale : astropy.Quantity, optional
        Outer scale of turbulence (L0) in meters. Above this scale, turbulence energy 
        saturates. Default: None (infinite outer scale).
    
    name : str, optional
        Name of the atmosphere layer for identification in diagrams
    
    Notes
    -----
    - The phase screen is generated in Fourier space using Kolmogorov statistics (f^-11/3 PSD).
    - Time evolution: screen drifts at wind_speed, so screen(t) = screen(x - v*t, y).
    - Chromatic behavior: phase(λ) = 2π * OPD / λ, so shorter wavelengths see larger phase.
    
    Examples
    --------
    >>> # Good seeing: 100nm RMS OPD, 5 m/s wind
    >>> atm = Atmosphere(rms=100*u.nm, wind_speed=5*u.m/u.s, seed=42)
    >>> 
    >>> # Poor seeing: 500nm RMS OPD, strong wind
    >>> atm = Atmosphere(rms=500*u.nm, wind_speed=15*u.m/u.s)
    >>> 
    >>> # Custom wind direction (northeast)
    >>> atm = Atmosphere(rms=200*u.nm, wind_speed=10*u.m/u.s, wind_direction=45)
    """
    def __init__(self, 
                 rms: u.Quantity = 100*u.nm,
                 wind_speed: Union[u.Quantity, Tuple[u.Quantity, u.Quantity]] = 5*u.m/u.s,
                 wind_direction: float = 0.0,
                 seed: Optional[int] = None,
                 inner_scale: Optional[u.Quantity] = None,
                 outer_scale: Optional[u.Quantity] = None,
                 name: Optional[str] = None):
        super().__init__(name=name or "Atmosphere")
        
        # Store OPD RMS in meters (enforce Quantity)
        self.rms = float(u.Quantity(rms, u.m).to(u.m).value)
        
        # Parse wind velocity (enforce Quantity)
        if isinstance(wind_speed, tuple):
            # (vx, vy) components provided
            vx = u.Quantity(wind_speed[0], u.m/u.s).to(u.m/u.s).value
            vy = u.Quantity(wind_speed[1], u.m/u.s).to(u.m/u.s).value
            self.wind_velocity = np.array([vx, vy], dtype=float)
        else:
            # Scalar speed + direction
            speed = u.Quantity(wind_speed, u.m/u.s).to(u.m/u.s).value
            angle_rad = np.deg2rad(wind_direction)
            self.wind_velocity = np.array([speed * np.cos(angle_rad), 
                                          speed * np.sin(angle_rad)], dtype=float)
        
        # Seed for reproducibility
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        
        # Turbulence scales in meters (optional, for Von Karman model)
        self.inner_scale = None
        self.outer_scale = None
        if inner_scale is not None:
            self.inner_scale = float(u.Quantity(inner_scale, u.m).to(u.m).value)
        if outer_scale is not None:
            self.outer_scale = float(u.Quantity(outer_scale, u.m).to(u.m).value)
        
        # Cache for frozen turbulent screen
        self._frozen_screen = None
        self._screen_size = None


    def _generate_frozen_screen(self, N: int, oversample: int = 2) -> np.ndarray:
        """Generate a large frozen turbulent screen for temporal evolution.
        
        The screen is oversampled to allow smooth translation without aliasing.
        
        Parameters
        ----------
        N : int
            Base array size (will be multiplied by oversample)
        oversample : int
            Oversampling factor for smooth translation (default: 2)
        
        Returns
        -------
        screen : ndarray
            OPD screen in meters, shape (N*oversample, N*oversample)
        """
        Nlarge = N * oversample
        rng = np.random.default_rng(self.seed)
        
        # Generate frequency grid (cycles per pixel)
        fx = np.fft.fftfreq(Nlarge)
        fy = fx.copy()
        fxg, fyg = np.meshgrid(fx, fy)
        f = np.sqrt(fxg ** 2 + fyg ** 2)

        # Avoid zero frequency singularity
        nonzero = f[f > 0]
        if nonzero.size == 0:
            fmin = 1.0 / float(Nlarge)
        else:
            fmin = float(nonzero.min())
        f[0, 0] = fmin

        # Kolmogorov filter amplitude ~ f^{-11/6} (sqrt of PSD ~ f^{-11/3})
        with np.errstate(divide='ignore', invalid='ignore'):
            filt = f ** (-11.0 / 6.0)
        
        # Apply Von Karman modifications if scales are specified
        if self.inner_scale is not None or self.outer_scale is not None:
            # Convert frequency to spatial scale (in pixels)
            # For inner scale cutoff: high-pass filter
            if self.inner_scale is not None:
                # inner scale in pixels (assuming screen spans self.diameter)
                # This is approximate - proper implementation needs pupil diameter info
                pass  # TODO: implement Von Karman inner scale
            
            if self.outer_scale is not None:
                # outer scale cutoff: low-pass filter
                pass  # TODO: implement Von Karman outer scale
        
        # Cap extreme values
        filt = np.nan_to_num(filt, nan=filt.max(), posinf=filt.max(), neginf=0.0)

        # Generate complex Gaussian white noise in Fourier domain
        real = rng.normal(size=(Nlarge, Nlarge))
        imag = rng.normal(size=(Nlarge, Nlarge))
        fourier = (real + 1j * imag) * filt

        # Zero DC component
        fourier[0, 0] = 0.0 + 0.0j

        # Enforce Hermitian symmetry for real-valued output
        fourier = (fourier + np.conj(np.flipud(np.fliplr(fourier)))) / 2.0

        # Inverse FFT to get OPD screen (real-valued, in arbitrary units)
        opd_screen = np.fft.ifft2(fourier).real

        # Normalize to requested RMS (in meters)
        screen_rms = float(np.std(opd_screen))
        if screen_rms <= 0 or not np.isfinite(screen_rms):
            return np.zeros((Nlarge, Nlarge), dtype=float)
        
        opd_screen = opd_screen / screen_rms * float(self.rms)
        return opd_screen

    def _extract_screen_at_time(self, time: u.Quantity, N: int) -> np.ndarray:
        """Extract N×N screen from frozen turbulence at given time.
        
        Uses bilinear interpolation to extract a shifted window from the frozen screen.
        
        Parameters
        ----------
        time : astropy.Quantity
            Observation time (shift = wind_velocity * time)
        N : int
            Output array size
        
        Returns
        -------
        screen : ndarray
            OPD screen in meters, shape (N, N)
        """
        # Ensure frozen screen exists
        if self._frozen_screen is None or self._screen_size != N:
            self._frozen_screen = self._generate_frozen_screen(N, oversample=2)
            self._screen_size = N
        
        # Convert time to shift in pixels
        if hasattr(time, 'to'):
            time_s = time.to(u.s).value
        else:
            time_s = u.Quantity(time, u.s).to(u.s).value
        
        # Compute shift in meters: displacement = velocity * time
        # Assume screen pixel size ~ diameter / N (approximate)
        # For now, use normalized shift (shift in units of N)
        # More robust: shift_pixels = (wind_velocity * time) / pixel_physical_size
        # Simple approach: shift in fraction of array size
        Nlarge = self._frozen_screen.shape[0]
        
        # Shift in pixels (assume screen spans ~2*diameter to allow drift)
        # Pixel size in frozen screen: diameter / N (base resolution)
        # For simplicity: shift normalized to base array size
        shift_normalized = self.wind_velocity * time_s / (N * 0.1)  # heuristic scaling
        shift_pixels = shift_normalized * N
        
        # Extract shifted window using np.roll (periodic boundaries)
        # Roll in both x and y
        shifted = np.roll(self._frozen_screen, 
                         shift=(-int(shift_pixels[0]), -int(shift_pixels[1])), 
                         axis=(1, 0))
        
        # Extract central N×N region
        Nlarge = shifted.shape[0]
        start = (Nlarge - N) // 2
        end = start + N
        screen = shifted[start:end, start:end]
        
        return screen

    def process(self, wavefront: Union[Wavefront, List[Wavefront]], context: Context = None) -> Union[Wavefront, WavefrontArray]:
        """Apply atmospheric turbulence to wavefront.
        
        Converts OPD (optical path difference) to phase: φ = 2π * OPD / λ
        This makes the aberration chromatic - shorter wavelengths see larger phase shifts.
        
        If a downstream TelescopeArray is detected in the context, this method can
        optimize processing by splitting the wavefront into multiple patches (one per
        collector) instead of processing a single huge wavefront.
        
        Parameters
        ----------
        wavefront : Wavefront or List[Wavefront]
            Input wavefront(s).
        context : Context, optional
            Simulation context (may contain time information)
        
        Returns
        -------
        wavefront : Wavefront or WavefrontArray
            Wavefront(s) with atmospheric phase applied
        """
        if wavefront is None:
            return None

        # 1. Check for downstream TelescopeArray to optimize
        target_collectors = []
        if context is not None and hasattr(context, 'layers'):
            # Find this atmosphere layer index
            my_index = -1
            for i, layer in enumerate(context.layers):
                if layer is self:
                    my_index = i
                    break
            
            # Look for next TelescopeArray
            if my_index != -1:
                for i in range(my_index + 1, len(context.layers)):
                    layer = context.layers[i]
                    if type(layer).__name__ == 'TelescopeArray':
                        target_collectors = layer.collectors
                        break
        
        # Check if we should use optimization (multiple collectors found)
        # And input is a single Wavefront (not already split)
        use_optimization = (len(target_collectors) > 0 and 
                           not isinstance(wavefront, (list, WavefrontArray)))
        
        if use_optimization:
            return self._process_optimized(wavefront, target_collectors, context)

        # Standard processing (single wavefront)
        try:
            if wavefront.ndim == 3:
                N = wavefront.shape[-1]
            else:
                N = wavefront.shape[0]
        except Exception:
            return wavefront

        # Get observation time from context or default to t=0
        if context is not None and hasattr(context, 'time'):
            time = context.time
        else:
            time = 0.0 * u.s
        
        # Extract OPD screen at this time (frozen-flow evolution)
        opd_screen = self._extract_screen_at_time(time, N)
        
        # Convert OPD to phase: φ = 2π * OPD / λ
        # wavefront.wavelength should be in meters
        if hasattr(wavefront, 'wavelength') and wavefront.wavelength is not None:
            wavelength_m = u.Quantity(wavefront.wavelength, u.m).to(u.m).value
        else:
            # Default wavelength if not specified (550 nm, visible)
            wavelength_m = 550e-9
        
        # Phase in radians
        phase = 2.0 * np.pi * opd_screen / wavelength_m
        
        # Apply phase screen (pure phase modulation)
        wavefront[:] = wavefront * np.exp(1j * phase).astype(wavefront.dtype)
        return wavefront

    def _process_optimized(self, wavefront: Wavefront, collectors: List['Collector'], context: Context) -> WavefrontArray:
        """Optimized processing for telescope arrays."""
        if wavefront.ndim == 3:
            N_in = wavefront.shape[-1]
        else:
            N_in = wavefront.shape[0]
        
        # Get time
        if context is not None and hasattr(context, 'time'):
            time = context.time
        else:
            time = 0.0 * u.s
        time_s = time.to(u.s).value if hasattr(time, 'to') else float(time)
            
        # 1. Determine geometry
        # Assume all collectors have roughly same size for pixel scale calculation
        # Use the first one as reference
        ref_size = collectors[0].size
        ref_size_m = ref_size.to(u.m).value if hasattr(ref_size, 'to') else float(ref_size)
        
        # Pixel scale required to match input wavefront resolution
        pixel_scale_m = ref_size_m / N_in
        
        # Find bounding box of array
        positions = np.array([c.position for c in collectors])
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        # Add padding for collector size
        padding = ref_size_m / 2.0
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        # Array dimensions
        width_m = max_x - min_x
        height_m = max_y - min_y
        
        # Required screen size in pixels
        W_pix = int(np.ceil(width_m / pixel_scale_m))
        H_pix = int(np.ceil(height_m / pixel_scale_m))
        N_global = max(W_pix, H_pix)
        
        # Limit global screen size to avoid memory explosion
        MAX_N = 4096
        if N_global > MAX_N:
            # We must downsample the global screen
            # This means we lose high-frequency correlations, but keep piston/tip-tilt
            scale_factor = MAX_N / N_global
            N_global = MAX_N
            effective_pixel_scale = pixel_scale_m / scale_factor
            # print(f"Warning: Atmosphere screen downsampled by {1/scale_factor:.1f}x (Global N={N_global})")
        else:
            effective_pixel_scale = pixel_scale_m
            
        # 2. Generate global frozen screen (if needed)
        # We use the existing _generate_frozen_screen but we need to manage the size
        # The existing method caches based on N.
        # If we change N, it regenerates.
        
        # We need to ensure the screen covers the array + wind drift
        # _generate_frozen_screen generates N*oversample.
        # Let's use N_global.
        
        # Note: _extract_screen_at_time assumes the screen corresponds to the requested N.
        # Here we are doing manual extraction.
        
        if self._frozen_screen is None or self._screen_size != N_global:
            self._frozen_screen = self._generate_frozen_screen(N_global, oversample=2)
            self._screen_size = N_global
            
        # 3. Extract patches for each collector
        patches = []
        
        # Wind shift in pixels
        # shift_m = velocity * time
        shift_m = self.wind_velocity * time_s
        shift_pix_x = shift_m[0] / effective_pixel_scale
        shift_pix_y = shift_m[1] / effective_pixel_scale
        
        # The frozen screen is N_global * 2 (oversample=2)
        # Center of frozen screen corresponds to (0,0) of the array?
        # Or (0,0) of the generated screen?
        # Usually FFT screen is periodic.
        # Let's assume the screen covers the bounding box centered.
        
        screen_center_pix = self._frozen_screen.shape[0] // 2
        
        # Center of the array bounding box relative to (0,0) world
        bbox_center_x = (min_x + max_x) / 2.0
        bbox_center_y = (min_y + max_y) / 2.0
        
        for col in collectors:
            # Collector position relative to array bounding box center
            rel_x = col.position[0] - bbox_center_x
            rel_y = col.position[1] - bbox_center_y
            
            # Convert to pixels in the screen
            # Note: we subtract wind shift because screen moves? 
            # Taylor: Phase(x, t) = Screen(x - v*t)
            # So we sample at x - v*t
            
            sample_x_m = rel_x - shift_m[0]
            sample_y_m = rel_y - shift_m[1]
            
            sample_pix_x = int(sample_x_m / effective_pixel_scale)
            sample_pix_y = int(sample_y_m / effective_pixel_scale)
            
            # Center in array
            center_x = screen_center_pix + sample_pix_x
            center_y = screen_center_pix + sample_pix_y
            
            # Extract patch
            # We need N_in pixels if scale matches, or scaled amount
            patch_size_pix = int(ref_size_m / effective_pixel_scale)
            
            start_x = center_x - patch_size_pix // 2
            start_y = center_y - patch_size_pix // 2
            end_x = start_x + patch_size_pix
            end_y = start_y + patch_size_pix
            
            # Handle boundaries (periodic wrap)
            # Simple way: np.roll then slice
            # Efficient way: slice with wrap
            
            # For simplicity, let's use roll on the whole screen? No, slow.
            # Just handle indices.
            rows = np.arange(start_y, end_y) % self._frozen_screen.shape[0]
            cols = np.arange(start_x, end_x) % self._frozen_screen.shape[1]
            
            patch = self._frozen_screen[np.ix_(rows, cols)]
            
            # Resize to N_in if needed (if we downsampled)
            if patch.shape[0] != N_in:
                from scipy.ndimage import zoom
                zoom_factor = N_in / patch.shape[0]
                patch = zoom(patch, zoom_factor, order=1)
            
            patches.append(patch)
            
        # 4. Create output WavefrontArray
        output_wfs = []
        
        # Wavelength
        if hasattr(wavefront, 'wavelength') and wavefront.wavelength is not None:
            if hasattr(wavefront.wavelength, 'to'):
                wavelength_m = wavefront.wavelength.to(u.m).value
            else:
                wavelength_m = float(wavefront.wavelength)
        else:
            wavelength_m = 550e-9
            
        for patch in patches:
            # Create copy of input wavefront
            new_wf = wavefront.copy()
            
            # Convert OPD to phase
            phase = 2.0 * np.pi * patch / wavelength_m
            
            # Apply phase
            new_wf[:] = new_wf * np.exp(1j * phase).astype(new_wf.dtype)
            output_wfs.append(new_wf)
            
        return WavefrontArray(output_wfs)

    def plot_screen_animation(self,
                             collectors: Optional[Union['Collectors', 'TelescopeArray', List['Collectors']]] = None,
                             times: Optional[np.ndarray] = None,
                             wavelength: u.Quantity = 550e-9*u.m,
                             npix: int = 512,
                             fps: int = 10,
                             duration: Optional[u.Quantity] = None,
                             filename: Optional[str] = None,
                             show_colorbar: bool = True,
                             figsize: Tuple[float, float] = (10, 10)):
        """Create animation of atmospheric phase screen with optional collector overlays.
        
        Displays the temporal evolution of the turbulent phase screen with frozen-flow,
        optionally overlaying collector apertures. The screen extent automatically adjusts
        to show all collectors with 20% margin.
        
        Parameters
        ----------
        collectors : Collectors, TelescopeArray, list of Collectors, or None
            Collector configuration(s) to overlay. If None, shows phase screen only.
            - Single Collectors: one telescope aperture
            - TelescopeArray: all baseline-separated apertures
            - List of Collectors: multiple independent telescopes
        times : ndarray, optional
            Observation times in seconds. Auto-generated if None.
        wavelength : astropy.Quantity
            Wavelength for phase calculation (default: 550nm).
        npix : int
            Phase screen resolution (default: 512).
        fps : int
            Animation frame rate (default: 10).
        duration : astropy.Quantity, optional
            Total duration in seconds (default: 5s).
        filename : str, optional
            Save path for animation (requires ffmpeg/pillow).
        show_colorbar : bool
            Show phase colorbar.
        figsize : Tuple[float, float]
            Figure size in inches.
        
        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            Animation object.
        
        Examples
        --------
        >>> # Phase screen only
        >>> atm = Atmosphere(rms=100*u.nm, wind_speed=10*u.m/u.s)
        >>> anim = atm.plot_screen_animation()
        >>> 
        >>> # With interferometer
        >>> vlti = TelescopeArray.vlti()
        >>> anim = atm.plot_screen_animation(collectors=vlti, duration=3*u.s)
        """
        from matplotlib.animation import FuncAnimation
        
        # Parse duration
        if duration is None:
            duration = 5.0 * u.s
        duration_s = duration.to(u.s).value if hasattr(duration, 'to') else float(duration)
        
        # Generate time array
        if times is None:
            n_frames = int(fps * duration_s)
            times = np.linspace(0, duration_s, n_frames)
        else:
            times = np.asarray(times)
        
        # Parse wavelength
        wavelength_m = wavelength.to(u.m).value if hasattr(wavelength, 'to') else float(wavelength)
        
        # Extract collector list from input
        collector_list = []
        array_name = "Atmospheric Phase Screen"
        
        if collectors is not None:
            if isinstance(collectors, TelescopeArray):
                collector_list = collectors.collectors
                array_name = f"{collectors.name}"
            elif isinstance(collectors, list):
                for c_obj in collectors:
                    if hasattr(c_obj, 'collectors'):
                        collector_list.extend(c_obj.collectors)
                array_name = f"{len(collector_list)} collectors"
            elif hasattr(collectors, 'collectors'):
                # Single Collectors object
                collector_list = collectors.collectors
                array_name = "Collectors"
        
        # Determine screen extent based on collectors (with 20% margin)
        if len(collector_list) > 0:
            # Find bounding box of all collectors
            min_x, max_x = 0, 0
            min_y, max_y = 0, 0
            
            for col in collector_list:
                pos = col.position
                size = col.size
                size_m = size.to(u.m).value if hasattr(size, 'to') else float(size)
                radius = size_m / 2.0
                
                min_x = min(min_x, pos[0] - radius)
                max_x = max(max_x, pos[0] + radius)
                min_y = min(min_y, pos[1] - radius)
                max_y = max(max_y, pos[1] + radius)
            
            # Add 20% margin
            width = max_x - min_x
            height = max_y - min_y
            margin_x = width * 0.2
            margin_y = height * 0.2
            
            extent_x = [min_x - margin_x, max_x + margin_x]
            extent_y = [min_y - margin_y, max_y + margin_y]
            
            # Make square extent (use max dimension)
            max_dim = max(extent_x[1] - extent_x[0], extent_y[1] - extent_y[0])
            center_x = (extent_x[0] + extent_x[1]) / 2.0
            center_y = (extent_y[0] + extent_y[1]) / 2.0
            
            extent = [center_x - max_dim/2, center_x + max_dim/2,
                     center_y - max_dim/2, center_y + max_dim/2]
        else:
            # No collectors: use default extent
            default_extent = 10.0  # meters
            extent = [-default_extent, default_extent, -default_extent, default_extent]
        
        # Create figure
        fig, ax = _plt.subplots(figsize=figsize)
        
        # Mock context for time evolution
        class TimeContext:
            def __init__(self, t):
                self.time = t * u.s
        
        # Generate initial phase screen
        wf_init = Wavefront(wavelength=wavelength, size=npix)
        wf_init[:] = np.ones((npix, npix), dtype=np.complex128)
        ctx_init = TimeContext(times[0])
        wf_atm_init = self.process(wf_init, ctx_init)
        phase_init = np.angle(wf_atm_init)
        
        # Display initial phase screen
        im = ax.imshow(phase_init, origin='lower', cmap='twilight',
                      extent=extent, vmin=-np.pi, vmax=np.pi,
                      interpolation='bilinear')
        
        # Overlay collector apertures
        for col in collector_list:
            pupil = col.pupil
            if pupil is None:
                continue
            
            pos = col.position
            
            # Render pupil at high resolution
            npix_pupil = 256
            pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
            
            # Create RGBA overlay (white with transparency)
            overlay = np.zeros((npix_pupil, npix_pupil, 4), dtype=float)
            overlay[..., :3] = 1.0  # white
            overlay[..., 3] = pupil_arr * 0.8  # alpha
            
            # Physical extent of this pupil
            diam = pupil.diameter
            extent_pupil = [pos[0] - diam/2, pos[0] + diam/2,
                           pos[1] - diam/2, pos[1] + diam/2]
            
            ax.imshow(overlay, origin='lower', extent=extent_pupil,
                     zorder=10, interpolation='bilinear')
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        
        title = ax.set_title(
            f'{array_name}\\n'
            f't={times[0]:.2f}s, λ={wavelength_m*1e9:.0f}nm, '
            f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s'
        )
        
        if show_colorbar:
            _plt.colorbar(im, ax=ax, label='Phase (radians)', fraction=0.046, pad=0.04)
        
        # Animation update function
        def update(frame_idx):
            t = times[frame_idx]
            
            # Generate phase screen at time t
            wf = Wavefront(wavelength=wavelength, size=npix)
            wf[:] = np.ones((npix, npix), dtype=np.complex128)
            ctx = TimeContext(t)
            wf_atm = self.process(wf, ctx)
            phase = np.angle(wf_atm)
            
            # Update phase screen
            im.set_data(phase)
            
            # Update title
            title.set_text(
                f'{array_name}\\n'
                f't={t:.2f}s, λ={wavelength_m*1e9:.0f}nm, '
                f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s'
            )
            
            return [im, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(times),
                           interval=1000.0/fps, blit=False, repeat=True)
        
        # Save if requested
        if filename is not None:
            try:
                anim.save(filename, fps=fps, dpi=100)
                print(f"Animation saved to {filename}")
            except Exception as e:
                print(f"Warning: Could not save animation: {e}")
        
        _plt.tight_layout()
        return anim

    def plot_animation(self, 
                      collectors: Union['Collectors', 'TelescopeArray', List['Collectors']], 
                      times: Optional[np.ndarray] = None,
                      wavelength: u.Quantity = 550e-9*u.m,
                      npix: int = 512,
                      fps: int = 10,
                      duration: Optional[u.Quantity] = None,
                      filename: Optional[str] = None,
                      show_colorbar: bool = True,
                      figsize: Tuple[float, float] = (8, 8)):
        """Create an animation of atmospheric phase screen with collectors overlay.
        
        Shows the temporal evolution of the turbulent phase screen as it drifts
        with the wind (frozen-flow), with the aperture geometry of collectors
        superimposed for reference.
        
        Parameters
        ----------
        collectors : Collectors, Interferometer, or list of Collectors
            Collector configuration(s) to overlay on the phase screen.
            - Single Collectors instance: shows one telescope aperture
            - Interferometer: shows all baseline-separated apertures
            - List of Collectors: shows multiple independent telescopes
        times : ndarray, optional
            Array of observation times in seconds. If None, generates evenly
            spaced times from 0 to duration.
        wavelength : astropy.Quantity
            Wavelength for phase calculation (default: 550nm visible).
        npix : int
            Resolution of the phase screen array (default: 512).
        fps : int
            Frames per second for the animation (default: 10).
        duration : astropy.Quantity, optional
            Total duration of the animation in seconds. If None, uses 5 seconds.
        filename : str, optional
            If provided, saves animation to this file (e.g., 'atm_animation.mp4').
            Requires ffmpeg or pillow for saving.
        show_colorbar : bool
            Whether to show colorbar for phase values.
        figsize : Tuple[float, float]
            Figure size in inches (width, height).
        
        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object. Call plt.show() to display.
        
        Examples
        --------
        >>> # Single telescope
        >>> atm = Atmosphere(rms=100*u.nm, wind_speed=10*u.m/u.s)
        >>> collectors = Collectors()
        >>> collectors.add(pupil=Pupil.like('VLT'), position=(0, 0), size=8*u.m)
        >>> anim = atm.plot_animation(collectors, duration=3*u.s)
        >>> plt.show()
        >>> 
        >>> # Interferometer array
        >>> interferometer = TelescopeArray.vlti()
        >>> anim = atm.plot_animation(interferometer, duration=5*u.s)
        """
        from matplotlib.animation import FuncAnimation
        
        # Parse duration
        if duration is None:
            duration = 5.0 * u.s
        duration_s = duration.to(u.s).value if hasattr(duration, 'to') else float(duration)
        
        # Generate time array if not provided
        if times is None:
            n_frames = int(fps * duration_s)
            times = np.linspace(0, duration_s, n_frames)
        else:
            times = np.asarray(times)
        
        # Parse wavelength
        wavelength_m = wavelength.to(u.m).value if hasattr(wavelength, 'to') else float(wavelength)
        
        # Normalize collectors input to a list
        if isinstance(collectors, TelescopeArray):
            # Extract aperture configuration from telescope array
            collector_list = collectors.collectors
            array_name = collectors.name
        elif isinstance(collectors, list):
            # List of Collectors objects
            collector_list = []
            for c_obj in collectors:
                if hasattr(c_obj, 'collectors'):
                    collector_list.extend(c_obj.collectors)
            array_name = f"{len(collector_list)} collectors"
        else:
            # Single Collectors object
            if hasattr(collectors, 'collectors'):
                collector_list = collectors.collectors
                array_name = "Collectors"
            else:
                raise TypeError("collectors must be Collectors, TelescopeArray, or list of Collectors")
        
        # Determine screen extent based on collectors (with 20% margin)
        if len(collector_list) > 0:
            # Find bounding box of all collectors
            min_x, max_x = 0, 0
            min_y, max_y = 0, 0
            
            for col in collector_list:
                pos = col.position
                size = col.size
                size_m = size.to(u.m).value if hasattr(size, 'to') else float(size)
                radius = size_m / 2.0
                
                min_x = min(min_x, pos[0] - radius)
                max_x = max(max_x, pos[0] + radius)
                min_y = min(min_y, pos[1] - radius)
                max_y = max(max_y, pos[1] + radius)
            
            # Add 20% margin
            width = max_x - min_x
            height = max_y - min_y
            margin_x = width * 0.2
            margin_y = height * 0.2
            
            extent_x = [min_x - margin_x, max_x + margin_x]
            extent_y = [min_y - margin_y, max_y + margin_y]
            
            # Make square extent (use max dimension)
            max_dim = max(extent_x[1] - extent_x[0], extent_y[1] - extent_y[0])
            center_x = (extent_x[0] + extent_x[1]) / 2.0
            center_y = (extent_y[0] + extent_y[1]) / 2.0
            
            extent = [center_x - max_dim/2, center_x + max_dim/2,
                     center_y - max_dim/2, center_y + max_dim/2]
        else:
            # No collectors: use default extent
            default_extent = 10.0  # meters
            extent = [-default_extent, default_extent, -default_extent, default_extent]
        
        # Create figure
        fig, ax = _plt.subplots(figsize=figsize)
        
        # Mock context for time evolution
        class TimeContext:
            def __init__(self, t):
                self.time = t * u.s
        
        # Initialize with first frame
        wf_init = Wavefront(wavelength=wavelength, npix=npix)
        wf_init[:] = np.ones((npix, npix), dtype=np.complex128)
        ctx_init = TimeContext(times[0])
        wf_atm_init = self.process(wf_init, ctx_init)
        phase_init = np.angle(wf_atm_init)
        # Squeeze to 2D if needed (wavefront has shape (samples, h, w))
        if phase_init.ndim == 3:
            phase_init = phase_init[0]
        
        # Plot initial phase screen
        im = ax.imshow(phase_init, origin='lower', cmap='twilight', 
                      extent=extent, vmin=-np.pi, vmax=np.pi, interpolation='bilinear')
        
        # Overlay collector apertures
        pupil_overlays = []
        for col in collector_list:
            pupil = col.pupil
            if pupil is None:
                continue
            
            pos = col.position
            
            # Render pupil at higher resolution for better visibility
            npix_pupil = 256
            pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
            
            # Create RGBA overlay (white aperture with transparency)
            overlay = np.zeros((npix_pupil, npix_pupil, 4), dtype=float)
            overlay[..., :3] = 1.0  # white
            overlay[..., 3] = pupil_arr * 0.7  # alpha channel (increased for better visibility)
            
            # Physical extent of pupil (convert diameter to float if Quantity)
            diam_m = pupil.diameter.to(u.m).value if isinstance(pupil.diameter, u.Quantity) else float(pupil.diameter)
            extent_pupil = [pos[0] - diam_m/2, pos[0] + diam_m/2, 
                           pos[1] - diam_m/2, pos[1] + diam_m/2]
            
            overlay_im = ax.imshow(overlay, origin='lower', extent=extent_pupil, 
                                  zorder=10, interpolation='bilinear')
            pupil_overlays.append(overlay_im)
        
        # Force the axis limits to the calculated extent (matplotlib auto-adjusts otherwise)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        
        title = ax.set_title(f'Atmospheric Phase Screen - {array_name}\\n' + 
                            f't={times[0]:.2f}s, λ={wavelength_m*1e9:.0f}nm, ' + 
                            f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s')
        
        if show_colorbar:
            cbar = _plt.colorbar(im, ax=ax, label='Phase (radians)', fraction=0.046, pad=0.04)
        
        # Animation update function
        def update(frame_idx):
            t = times[frame_idx]
            
            # Generate phase screen at time t
            wf = Wavefront(wavelength=wavelength, npix=npix)
            wf[:] = np.ones((npix, npix), dtype=np.complex128)
            ctx = TimeContext(t)
            wf_atm = self.process(wf, ctx)
            phase = np.angle(wf_atm)
            # Squeeze to 2D if needed
            if phase.ndim == 3:
                phase = phase[0]
            
            # Update image data
            im.set_data(phase)
            
            # Update title
            title.set_text(f'Atmospheric Phase Screen - {array_name}\\n' + 
                          f't={t:.2f}s, λ={wavelength_m*1e9:.0f}nm, ' + 
                          f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s')
            
            return [im, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(times), 
                           interval=1000.0/fps, blit=False, repeat=True)
        
        # Save if filename provided
        if filename is not None:
            try:
                anim.save(filename, fps=fps, dpi=100)
                print(f"Animation saved to {filename}")
            except Exception as e:
                print(f"Warning: Could not save animation: {e}")
        
        _plt.tight_layout()
        return anim





class AdaptiveOptics(Element):
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

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
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

