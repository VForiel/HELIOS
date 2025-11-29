"""Telescope array classes for single and interferometric observations.

This module provides classes for managing telescope configurations:
- Collector: Single telescope with pupil geometry and position (data object)
- TelescopeArray: Array of one or more telescopes with spatial positioning (Layer subclass)

TelescopeArray unifies single-telescope and interferometric observations:
- Single telescope: Add one collector at position (0, 0)
- Interferometer: Add multiple collectors at different baseline positions
"""
import numpy as np
from astropy import units as u
from typing import Tuple, Optional
import matplotlib.pyplot as _plt

from ..core.context import Layer, Context
from ..core.simulation import Wavefront
from .pupil import Pupil


class Collector:
    """Represents a single telescope/collector with pupil geometry and position.
    
    A Collector encapsulates the properties of an individual telescope aperture,
    including its pupil geometry (transmission pattern), physical size, and
    position in the aperture plane (for interferometric arrays).
    
    This class provides a cleaner, more object-oriented alternative to storing
    collector properties in dictionaries.
    
    Parameters
    ----------
    pupil : Pupil
        Pupil geometry defining the aperture transmission pattern.
    position : Tuple[float, float]
        (x, y) position in the aperture plane (meters). For single telescopes,
        use (0, 0). For interferometric arrays, specify baseline coordinates.
    size : astropy.Quantity, optional
        Diameter of the collector aperture. If None, inferred from pupil.diameter.
    name : str, optional
        Descriptive name for this collector (e.g., "UT1", "AT3").
    **metadata
        Additional metadata (e.g., mount type, coating, location).
    
    Attributes
    ----------
    pupil : Pupil
        The pupil geometry object.
    position : Tuple[float, float]
        Baseline coordinates in meters.
    size : astropy.Quantity
        Aperture diameter in meters.
    name : str
        Collector identifier.
    metadata : dict
        Additional properties.
    
    Examples
    --------
    >>> # Create a VLT UT collector
    >>> pupil_vlt = Pupil.like('VLT')
    >>> ut1 = Collector(pupil=pupil_vlt, position=(0, 0), size=8.2*u.m, name="UT1")
    >>> print(ut1.name, ut1.size)
    UT1 8.2 m
    """
    def __init__(self, pupil: Pupil, position: Tuple[float, float] = (0, 0),
                 size: Optional[u.Quantity] = None, name: Optional[str] = None,
                 **metadata):
        self.pupil = pupil
        self.position = tuple(position)
        
        # Infer size from pupil if not provided
        if size is None:
            if hasattr(pupil, 'diameter'):
                size = pupil.diameter * u.m  # pupil.diameter stored as float in meters
            else:
                size = 1.0 * u.m
        self.size = size
        
        self.name = name or f"Collector@({position[0]:.1f},{position[1]:.1f})"
        self.metadata = metadata
    
    def __repr__(self):
        return f"Collector(name='{self.name}', position={self.position}, size={self.size})"


class TelescopeArray(Layer):
    """Array of one or more telescopes with pupil geometries and spatial positioning.
    
    This class unifies single-telescope and interferometric observations by managing
    an array of collectors with arbitrary spatial positions. It handles both:
    
    **Single telescope**: Add one collector at position (0, 0)
        - Used for conventional single-aperture observations
        - The pupil mask is applied at the center of the wavefront
        
    **Interferometer**: Add multiple collectors at different positions
        - Used for interferometric imaging with spatially separated apertures
        - Each pupil is positioned at its baseline coordinates (u,v plane)
        - Enables aperture synthesis and high angular resolution
    
    The spatial positioning is automatically handled: collectors at (0,0) are 
    treated as co-located, while non-zero positions create a dilute aperture array.
    
    Parameters
    ----------
    name : str, optional
        Name of the telescope configuration (e.g., "VLT-UT4", "VLTI", "CHARA").
    latitude : astropy.Quantity, optional
        Geographic latitude of the observatory (degrees).
    longitude : astropy.Quantity, optional
        Geographic longitude of the observatory (degrees).
    altitude : astropy.Quantity, optional
        Altitude above sea level (meters).
    
    Attributes
    ----------
    collectors : List[Collector]
        List of Collector objects, each with pupil, position, and metadata.
    name : str
        Configuration name.
    latitude, longitude, altitude : astropy.Quantity
        Observatory geographic coordinates.
    
    Examples
    --------
    >>> # Single telescope (VLT UT4)
    >>> vlt = TelescopeArray(name="VLT-UT4", latitude=-24.6*u.deg, altitude=2635*u.m)
    >>> pupil_vlt = helios.Pupil.vlt()
    >>> vlt.add_collector(pupil=pupil_vlt, position=(0, 0), size=8.2*u.m)
    >>> 
    >>> # Interferometer (VLTI with 4 UTs)
    >>> vlti = TelescopeArray(name="VLTI")
    >>> for i, pos in enumerate([(0,0), (47,0), (47,47), (0,47)]):
    >>>     vlti.add_collector(pupil=pupil_vlt, position=pos, size=8.2*u.m, name=f"UT{i+1}")
    >>> 
    >>> # Check if this is interferometric (multiple non-colocated apertures)
    >>> print(f"Interferometric: {vlti.is_interferometric()}")
    """
    
    def __init__(self, name: Optional[str] = None,
                 latitude: u.Quantity = 0*u.deg, 
                 longitude: u.Quantity = 0*u.deg, 
                 altitude: u.Quantity = 0*u.m):
        super().__init__()
        self.name = name or "TelescopeArray"
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.collectors = []
    
    def add_collector(self, pupil: Pupil, position: Tuple[float, float] = (0, 0), 
                     size: Optional[u.Quantity] = None, name: Optional[str] = None, **kwargs):
        """Add a collector to the telescope array.
        
        Parameters
        ----------
        pupil : Pupil
            Pupil geometry for this collector (defines aperture shape).
        position : Tuple[float, float], optional
            (x, y) baseline coordinates in meters. Default (0, 0) for single telescope.
            For interferometers, specify spatial separation between apertures.
        size : astropy.Quantity, optional
            Diameter of the collector. If None, inferred from pupil.diameter.
        name : str, optional
            Descriptive name for this collector (e.g., "UT1", "AT2").
        **kwargs
            Additional metadata (e.g., mount type, coating).
        
        Examples
        --------
        >>> array = TelescopeArray(name="CHARA")
        >>> pupil = helios.Pupil(diameter=1*u.m)
        >>> array.add_collector(pupil, position=(0, 0), size=1*u.m, name="S1")
        >>> array.add_collector(pupil, position=(100, 0), size=1*u.m, name="S2")
        """
        collector = Collector(pupil=pupil, position=position, size=size, name=name, **kwargs)
        self.collectors.append(collector)
    
    def is_interferometric(self) -> bool:
        """Check if this array has multiple non-colocated apertures (interferometric).
        
        Returns True if there are multiple collectors at different positions,
        False for single telescope or all collectors at (0, 0).
        """
        if len(self.collectors) <= 1:
            return False
        positions = {c.position for c in self.collectors}
        return len(positions) > 1
    
    def get_baseline_array(self) -> np.ndarray:
        """Return array of baseline vectors (u,v coordinates) in meters.
        
        Returns
        -------
        baselines : ndarray
            Array of shape (N, 2) where N is the number of collectors.
            Each row is (x, y) position in meters.
        """
        return np.array([c.position for c in self.collectors], dtype=float)
    
    def plot_array(self, ax: Optional[_plt.Axes] = None, show_pupils: bool = True,
                  pupil_scale: float = 1.0) -> _plt.Axes:
        """Plot the telescope array configuration.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_pupils : bool
            If True, render individual pupil shapes at each baseline position.
        pupil_scale : float
            Scale factor for pupil rendering (1.0 = actual size).
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            fig, ax = _plt.subplots(figsize=(8, 8))
        
        baselines = self.get_baseline_array()
        
        if show_pupils and len(self.collectors) > 0:
            # Determine plot extent and resolution
            max_extent = 0
            for collector in self.collectors:
                x, y = collector.position
                size = collector.size.to(u.m).value if hasattr(collector.size, 'to') else float(collector.size)
                max_extent = max(max_extent, abs(x) + size, abs(y) + size)
            
            # Create a large canvas to hold all pupils
            margin = max_extent * 0.15  # 15% margin
            canvas_size = 2 * (max_extent + margin)
            npix_canvas = int(canvas_size * 10)  # 10 pixels/meter
            canvas = np.zeros((npix_canvas, npix_canvas), dtype=float)
            
            # Pixel scale (meters per pixel)
            pixel_scale = canvas_size / npix_canvas
            
            # Render each pupil onto the canvas
            for collector in self.collectors:
                pupil = collector.pupil
                x_pos, y_pos = collector.position
                
                # Render pupil at appropriate resolution
                diam = pupil.diameter * pupil_scale
                npix_pupil = int(diam / pixel_scale)
                npix_pupil = max(32, min(npix_pupil, 256))  # Clamp to reasonable range
                pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
                
                # Calculate pixel position on canvas
                center_m = max_extent + margin
                x_pix = int((x_pos + center_m) / pixel_scale)
                y_pix = int((y_pos + center_m) / pixel_scale)
                
                # Half-size in pixels
                half_npix = npix_pupil // 2
                
                # Insert pupil into canvas
                x_start = max(0, x_pix - half_npix)
                x_end = min(npix_canvas, x_pix + half_npix)
                y_start = max(0, y_pix - half_npix)
                y_end = min(npix_canvas, y_pix + half_npix)
                
                # Corresponding slice in pupil array
                px_start = max(0, -(x_pix - half_npix))
                px_end = px_start + (x_end - x_start)
                py_start = max(0, -(y_pix - half_npix))
                py_end = py_start + (y_end - y_start)
                
                # Overlay pupil (take maximum to avoid overwriting)
                canvas[y_start:y_end, x_start:x_end] = np.maximum(
                    canvas[y_start:y_end, x_start:x_end],
                    pupil_arr[py_start:py_end, px_start:px_end]
                )
            
            # Display the full canvas
            extent = [-center_m, center_m, -center_m, center_m]
            ax.imshow(canvas, origin='lower', cmap='gray', extent=extent, alpha=0.9)
        else:
            # Simple scatter plot
            ax.scatter(baselines[:, 0], baselines[:, 1], s=100, c='blue', 
                      marker='o', edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Baseline x (m)')
        ax.set_ylabel('Baseline y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Title indicates if interferometric
        mode = "Interferometric" if self.is_interferometric() else "Single Telescope"
        ax.set_title(f'{self.name} - {mode} ({len(self.collectors)} collector{"s" if len(self.collectors) > 1 else ""})')
        
        return ax
    
    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        """Apply telescope array aperture mask to wavefront.
        
        This method handles both single-telescope and interferometric configurations:
        - If all collectors are at (0, 0): Pupils are combined multiplicatively (co-phased)
        - If collectors have different positions: Pupils are placed at their baseline positions
        
        For true interferometric fringe formation and beam combination, use dedicated
        photonics layers (PhotonicChip, TOPS, etc.) after this layer.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to process.
        context : Context
            Simulation context (unused in this implementation).
        
        Returns
        -------
        wavefront : Wavefront
            Wavefront with aperture mask applied.
        """
        try:
            N = wavefront.field.shape[0]
        except Exception:
            return wavefront
        
        if len(self.collectors) == 0:
            return wavefront
        
        # Determine if we need spatial positioning
        is_interferometric = self.is_interferometric()
        
        if not is_interferometric:
            # Single telescope or co-located: combine multiplicatively at center
            total_mask = np.ones((N, N), dtype=float)
            for collector in self.collectors:
                if isinstance(collector.pupil, Pupil):
                    try:
                        mask = collector.pupil.get_array(npix=N, soft=True)
                    except Exception:
                        mask = collector.pupil.get_array(npix=N, soft=False)
                    total_mask = total_mask * mask
            wavefront.field = wavefront.field * total_mask.astype(wavefront.field.dtype)
        else:
            # Interferometric: position each pupil at its baseline coordinates
            combined_mask = np.zeros((N, N), dtype=float)
            
            # Determine array extent
            baselines = self.get_baseline_array()
            max_extent = np.max(np.abs(baselines)) if len(baselines) > 0 else 1.0
            for collector in self.collectors:
                size_m = collector.size.to(u.m).value if hasattr(collector.size, 'to') else float(collector.size)
                max_extent = max(max_extent, size_m)
            
            # Pixel scale: meters per pixel
            pixel_scale = 2.0 * max_extent / float(N)
            
            # Render each collector pupil at its position
            for collector in self.collectors:
                pupil = collector.pupil
                x, y = collector.position
                
                # Render pupil
                pupil_arr = pupil.get_array(npix=N, soft=True)
                
                # Compute shift in pixels
                shift_x = int(x / pixel_scale)
                shift_y = int(y / pixel_scale)
                
                # Shift pupil to baseline position
                shifted_pupil = np.roll(pupil_arr, shift=(shift_y, shift_x), axis=(0, 1))
                
                # Add to combined mask
                combined_mask = np.maximum(combined_mask, shifted_pupil)
            
            # Apply to wavefront
            wavefront.field = wavefront.field * combined_mask.astype(wavefront.field.dtype)
        
        return wavefront


# Legacy aliases for backward compatibility
Telescope = TelescopeArray  # Single telescope is just TelescopeArray with one collector at (0,0)
Interferometer = TelescopeArray  # Interferometer is just TelescopeArray with multiple collectors at different positions
