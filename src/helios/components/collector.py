"""Telescope array classes for single and interferometric observations.

This module provides classes for managing telescope configurations:
- Telescope: Single telescope with pupil geometry and position (Component subclass)
- TelescopeArray: Array of one or more telescopes with spatial positioning (Component subclass)

TelescopeArray unifies single-telescope and interferometric observations:
- Single telescope: Add one telescope at position (0, 0)
- Interferometer: Add multiple telescopes at different baseline positions
"""
import numpy as np
from astropy import units as u
from typing import Tuple, Optional, Any, Union, List
import matplotlib.pyplot as _plt

from ..core.pipeline import Layer, SamplingLayer, Component, SamplingComponent, Pipeline
from ..utils.serialization import serialize_value, deserialize_value
from ..core.simulation import Wavefront, WavefrontArray
from .pupil import Pupil


class Telescope(SamplingComponent):
    __slots__ = ("pupil", "position", "size", "name")
    """Represents a single telescope with pupil geometry and position.
    
    A Telescope is a Component that encapsulates the properties of an individual
    telescope aperture, including its pupil geometry (transmission pattern), 
    physical size, and position in the aperture plane (for interferometric arrays).
    
    Telescopes are grouped within a TelescopeArray component for parallel processing
    in interferometric configurations or co-located single telescope observations.
    
    Parameters
    ----------
    pupil : Pupil
        Pupil geometry defining the aperture transmission pattern.
    position : Tuple[float, float] or Tuple[astropy.Quantity, astropy.Quantity]
        (x, y) position in the aperture plane. If floats, assumed to be meters.
        Can also be astropy Quantities. For single telescopes, use (0, 0).
        For interferometric arrays, specify baseline coordinates.
    size : astropy.Quantity, optional
        Diameter of the telescope aperture. If None, inferred from pupil.diameter.
    name : str, optional
        Descriptive name for this telescope (e.g., "UT1", "AT3").
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
        Telescope identifier (inherited from Component).
    metadata : dict
        Additional properties.
    
    Examples
    --------
    >>> # Create a VLT UT telescope
    >>> pupil_vlt = Pupil.like('VLT')
    >>> ut1 = Telescope(pupil=pupil_vlt, position=(0, 0), size=8.2*u.m, name="UT1")
    >>> print(ut1.name, ut1.size)
    UT1 8.2 m
    """
    def __init__(self, pupil: Pupil, position: Tuple[float, float] = (0, 0),
                 size: Optional[u.Quantity] = None, name: Optional[str] = None,
                 **metadata):
        # Handle position units (convert to meters if Quantity)
        x, y = position
        x = x.to(u.m).value if hasattr(x, 'to') else float(x)
        y = y.to(u.m).value if hasattr(y, 'to') else float(y)
        self.position = (x, y)

        # Initialize Component with name
        default_name = f"Telescope@({self.position[0]:.1f},{self.position[1]:.1f})"
        super().__init__(name=name or default_name)
        
        self.pupil = pupil
        
        # Infer size from pupil if not provided
        if size is None:
            if hasattr(pupil, 'diameter'):
                if isinstance(pupil.diameter, u.Quantity):
                    size = pupil.diameter
                else:
                    size = pupil.diameter * u.m
            else:
                size = 1.0 * u.m
        self.size = size
        
        self.metadata = metadata
        
    def to_dict(self) -> dict:
        """Serialize telescope."""
        data = super().to_dict()
        data.update({
            "pupil": self.pupil.to_dict(),
            "position": serialize_value(self.position),
            "size": serialize_value(self.size),
            "metadata": serialize_value(self.metadata)
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Telescope':
        """Create telescope from dict."""
        name = data.get("name")
        pupil_data = data.get("pupil")
        pupil = Pupil.from_dict(pupil_data) if pupil_data else None
        
        # Position can be list/tuple from JSON
        pos_raw = deserialize_value(data.get("position", (0,0)))
        # Ensure tuple
        if isinstance(pos_raw, list):
            pos_raw = tuple(pos_raw)
            
        size = deserialize_value(data.get("size"))
        metadata = deserialize_value(data.get("metadata", {}))
        
        return cls(pupil=pupil, position=pos_raw, size=size, name=name, **metadata)
    
    def process(self, wavefront: Wavefront, pipeline: Optional['Pipeline'] = None, auto_magnify: Optional[bool] = None) -> Any:
        """
        Process the wavefront through this telescope's pupil.
        
        Applies the pupil transmission pattern to the wavefront field.
        Also updates the wavefront's pixel scale to match the telescope size.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to process.
        pipeline : Pipeline, optional
            The simulation pipeline (unused in Telescope but required by Layer protocol).
        auto_magnify : bool, optional
            If True, resize wavefront to match telescope size.
            If False, crop wavefront to telescope size.
            If None, check sizes and warn if mismatch.
        
        Returns
        -------
        wavefront : Wavefront
            Wavefront with pupil mask applied and pixel scale updated.
        """
        # Backwards compatibility: if pipeline is passed as boolean, treat it as auto_magnify
        if isinstance(pipeline, bool) and auto_magnify is None:
            auto_magnify = pipeline
            pipeline = None
        
        # Handle auto_magnify logic
        if self.size is not None:
            # Ensure size is in meters
            telescope_size = self.size
            if not isinstance(telescope_size, u.Quantity):
                telescope_size = telescope_size * u.m
                
            wf_size = wavefront.width
            if not isinstance(wf_size, u.Quantity):
                wf_size = wf_size * u.m
            
            # Check if sizes match (with some tolerance)
            sizes_match = np.isclose(telescope_size.to(u.m).value, wf_size.to(u.m).value, rtol=1e-5)
            
            if auto_magnify is None:
                if not sizes_match:
                    import warnings
                    warnings.warn(f"Wavefront size ({wf_size}) does not match Telescope size ({telescope_size}). "
                                  f"Resizing wavefront metadata to match telescope (auto_magnify=True).")
                    auto_magnify = True
                else:
                    auto_magnify = False
            
            if auto_magnify:
                # Modify wavefront size metadata
                wavefront.width = telescope_size
                wavefront.pixel_scale = (telescope_size / wavefront.npix).to(u.m)
            else:
                # Crop wavefront
                wavefront = wavefront.crop(new_size=telescope_size, center=(0*u.m, 0*u.m))
        
        # Use the last dimension for spatial size (assuming square)
        # field shape is typically (samples, height, width) or just (height, width)
        N = wavefront.shape[-1]
        
        # Update pixel scale based on telescope size
        # The wavefront now represents the field at the pupil plane of this telescope
        if self.size is not None:
            # Ensure size is in meters
            size_m = self.size.to(u.m).value if hasattr(self.size, 'to') else float(self.size)
            wavefront.pixel_scale = (size_m / N) * u.m

        mask = self.pupil.get_array(npix=N, soft=True)
        wavefront[:] = wavefront * mask

        wavefront._last_focal_length_m = float(self.pupil.focal_length.to(u.m).value)

        return wavefront

        wavefront._last_focal_length_m = float(self.pupil.focal_length.to(u.m).value)

        return wavefront
    
    def __repr__(self):
        return f"Telescope(name='{self.name}', position={self.position}, size={self.size})"
    
    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for Telescope."""
        attrs = {}
        attrs['position'] = f"({self.position[0]:.2f}, {self.position[1]:.2f}) m"
        attrs['size'] = str(self.size)
        if hasattr(self.pupil, 'diameter'):
            attrs['pupil_diameter'] = f"{self.pupil.diameter:.2f} m"
        return attrs


class TelescopeArray(Telescope):
    __slots__ = ("positions", "latitude", "longitude", "altitude")
    """Array of identical telescopes at different spatial positions.
    
    Inherits from Telescope to share common attributes (pupil, size).
    Extends it by adding multiple positions and observatory coordinates.
    
    This class represents a telescope array where all telescopes share the same
    pupil geometry and size (inherited from Telescope), but are positioned at 
    different locations. This is the physical reality for most telescope arrays 
    (VLTI, LIFE, etc.).
    
    **Inheritance**:
    - Inherits from `Telescope`: shares `pupil`, `size`, `position` (set to first position)
    - Adds: `positions` (list of all positions), `latitude`, `longitude`, `altitude`
    
    **Single telescope**: One position at (0, 0)
        - Used for conventional single-aperture observations
        - The pupil mask is applied at the center of the wavefront
        
    **Interferometer**: Multiple positions at different coordinates
        - Used for interferometric imaging with spatially separated apertures
        - Each position receives a copy of the input wavefront
        - Enables aperture synthesis and high angular resolution
    
    Parameters
    ----------
    pupil : Pupil
        Shared pupil geometry for all telescopes in the array.
    size : astropy.Quantity
        Diameter of each telescope aperture.
    positions : List[Tuple[float, float]], optional
        List of (x, y) baseline coordinates in meters. Default [(0, 0)].
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
    pupil : Pupil
        Shared pupil geometry object (inherited from Telescope).
    size : astropy.Quantity
        Telescope aperture diameter (inherited from Telescope).
    position : Tuple[float, float]
        Position of first telescope (inherited from Telescope).
    positions : List[Tuple[float, float]]
        List of all telescope positions in meters.
    name : str
        Configuration name.
    latitude, longitude, altitude : astropy.Quantity
        Observatory geographic coordinates.
    
    Examples
    --------
    >>> # Single telescope (VLT UT4)
    >>> pupil_vlt = helios.Pupil.vlt()
    >>> vlt = TelescopeArray(pupil=pupil_vlt, size=8.2*u.m, positions=[(0, 0)], 
    ...                      name="VLT-UT4", latitude=-24.6*u.deg, altitude=2635*u.m)
    >>> 
    >>> # Interferometer (VLTI with 4 UTs)
    >>> positions = [(0,0), (47,0), (47,47), (0,47)]
    >>> vlti = TelescopeArray(pupil=pupil_vlt, size=8.2*u.m, positions=positions, name="VLTI")
    >>> 
    >>> # Check if this is interferometric (multiple non-colocated apertures)
    >>> print(f"Interferometric: {vlti.is_interferometric()}")
    """
    
    def __init__(self, pupil: Pupil, size: u.Quantity,
                 positions: Optional[List[Tuple[float, float]]] = None,
                 name: Optional[str] = None,
                 latitude: u.Quantity = 0*u.deg, 
                 longitude: u.Quantity = 0*u.deg, 
                 altitude: u.Quantity = 0*u.m):
        # Initialize parent Telescope with first position
        if positions is None:
            positions = [(0.0, 0.0)]
        first_position = positions[0]
        
        # Call parent __init__ with first position
        super().__init__(pupil=pupil, position=first_position, size=size, 
                        name=name or "TelescopeArray")
        
        # Add array-specific attributes
        self.positions = positions
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def to_dict(self) -> dict:
        """Serialize telescope array, extending parent Telescope serialization."""
        # Get parent serialization (includes pupil, size, position, name)
        data = super().to_dict()
        # Add array-specific attributes
        data.update({
            "positions": serialize_value(self.positions),
            "latitude": serialize_value(self.latitude),
            "longitude": serialize_value(self.longitude),
            "altitude": serialize_value(self.altitude)
        })
        return data

    @classmethod
    def from_dict(cls, data: dict, pipeline: Optional[Pipeline] = None) -> 'TelescopeArray':
        """Create telescope array from dict."""
        # Extract parent attributes
        name = data.get("name")
        pupil_data = data.get("pupil")
        pupil = Pupil.from_dict(pupil_data) if pupil_data else None
        size = deserialize_value(data.get("size"))
        
        # Extract array-specific attributes
        positions = deserialize_value(data.get("positions", [(0.0, 0.0)]))
        latitude = deserialize_value(data.get("latitude", 0*u.deg))
        longitude = deserialize_value(data.get("longitude", 0*u.deg))
        altitude = deserialize_value(data.get("altitude", 0*u.m))
        
        # Create array (parent __init__ will be called)
        array = cls(pupil=pupil, size=size, positions=positions, name=name,
                   latitude=latitude, longitude=longitude, altitude=altitude)
        array.metadata = data.get("metadata", {})
        return array
    
    @property
    def num_outputs(self) -> int:
        """Number of outputs produced by this array (one per telescope position)."""
        return len(self.positions)
    
    @num_outputs.setter
    def num_outputs(self, value):
        # Component.__init__ tries to set this, but we ignore it 
        # as it is dynamically derived from positions.
        pass
    
    @property
    def num_telescopes(self) -> int:
        """Number of telescopes in the array."""
        return len(self.positions)
    
    def add_position(self, x: float, y: float):
        """Add a telescope position to the array.
        
        Parameters
        ----------
        x : float
            X coordinate in meters.
        y : float
            Y coordinate in meters.
        
        Examples
        --------
        >>> array = TelescopeArray(pupil=pupil, size=8*u.m)
        >>> array.add_position(0, 0)
        >>> array.add_position(47, 0)
        """
        # Convert to float if Quantity
        if hasattr(x, 'to'):
            x = x.to(u.m).value
        if hasattr(y, 'to'):
            y = y.to(u.m).value
        self.positions.append((float(x), float(y)))
    
    def is_interferometric(self) -> bool:
        """Check if this array has multiple non-colocated apertures (interferometric).
        
        Returns True if there are multiple telescopes at different positions,
        False for single telescope or all telescopes at (0, 0).
        """
        if len(self.positions) <= 1:
            return False
        unique_positions = set(self.positions)
        return len(unique_positions) > 1
    
    def get_baseline_array(self) -> np.ndarray:
        """Return array of baseline vectors (u,v coordinates) in meters.
        
        Returns
        -------
        baselines : ndarray
            Array of shape (N, 2) where N is the number of telescopes.
            Each row is (x, y) position in meters.
        """
        return np.array(self.positions, dtype=float)
    
    def _get_detailed_attributes(self) -> dict:
        """Return detailed attributes for TelescopeArray."""
        attrs = {}
        attrs['num_telescopes'] = len(self.positions)
        attrs['pupil_diameter'] = str(self.size)
        if self.is_interferometric():
            attrs['configuration'] = "Interferometric"
            baselines = self.get_baseline_array()
            if len(baselines) > 1:
                max_baseline = np.max(np.linalg.norm(baselines - baselines[0], axis=1))
                attrs['max_baseline'] = f"{max_baseline:.2f} m"
        else:
            attrs['configuration'] = "Single telescope"
        if self.latitude != 0*u.deg or self.longitude != 0*u.deg:
            attrs['latitude'] = str(self.latitude)
            attrs['longitude'] = str(self.longitude)
        if self.altitude != 0*u.m:
            attrs['altitude'] = str(self.altitude)
        return attrs
    
    @classmethod
    def vlti(cls, uts: bool = True) -> 'TelescopeArray':
        """Create a VLTI (Very Large Telescope Interferometer) configuration.
        
        The VLTI at ESO Paranal Observatory consists of 4 Unit Telescopes (UTs) 
        or up to 4 Auxiliary Telescopes (ATs) that can be positioned on various 
        stations. This method creates a realistic configuration based on GPS 
        coordinates converted to baseline positions via tangent plane projection.
        
        Parameters
        ----------
        uts : bool, optional
            If True (default), create configuration with 4 Unit Telescopes (8.2m).
            If False, create configuration with 4 Auxiliary Telescopes (1.8m).
        
        Returns
        -------
        vlti : TelescopeArray
            VLTI interferometric array with 4 telescopes.
        
        Notes
        -----
        The baseline positions are derived from actual GPS coordinates of the 
        telescopes at Paranal:
        - GPS coordinates: longitude/latitude in degrees
        - Conversion: tangent plane projection with Earth radius + elevation (2635m)
        - Reference: PHISE project (https://github.com/VForiel/PHISE)
        
        UT configuration:
        - 4 Unit Telescopes of 8.2m diameter
        - Baselines ranging from ~47m to ~130m
        - Used for high-resolution interferometry
        
        AT configuration:
        - 4 Auxiliary Telescopes of 1.8m diameter  
        - Relocatable on a grid of stations
        - For this preset, we use a representative compact configuration
        
        Examples
        --------
        >>> # Create VLTI with Unit Telescopes
        >>> vlti_ut = TelescopeArray.vlti(uts=True)
        >>> print(f"VLTI UTs: {vlti_ut.num_telescopes} telescopes")
        >>> print(vlti_ut.get_baseline_array())
        
        >>> # Create VLTI with Auxiliary Telescopes
        >>> vlti_at = TelescopeArray.vlti(uts=False)
        >>> vlti_at.plot_array(show_pupils=True)
        """
        if uts:
            # VLTI Unit Telescopes (8.2m diameter)
            # Real baseline positions from GPS coordinates
            # Source: PHISE project telescope.py get_UT_telescopes()
            pupil = Pupil.like('VLT')
            diameter = 8.2 * u.m
            
            # Baseline positions (GPS → tangent plane projection)
            positions = [
                (-16.14, 62.74),   # UT1
                (0.00, 0.00),      # UT2 (reference)
                (63.03, 53.37),    # UT3
                (101.99, 34.54)    # UT4
            ]
            
            vlti = cls(pupil=pupil, size=diameter, positions=positions,
                      name="VLTI-UTs", latitude=-24.627*u.deg, 
                      longitude=-70.404*u.deg, altitude=2635*u.m)
        else:
            # VLTI Auxiliary Telescopes (1.8m diameter)
            # Representative compact configuration
            
            # Simple circular pupil for ATs
            pupil_at = Pupil(diameter=1.8*u.m)
            pupil_at.add_disk(radius=0.9*u.m)
            pupil_at.add_central_obscuration(diameter=0.2*u.m)
            diameter = 1.8 * u.m
            
            # Compact baseline configuration (example positions)
            positions = [
                (0.00, 0.00),      # AT1 (reference)
                (32.00, 0.00),     # AT2
                (16.00, 27.71),    # AT3
                (16.00, -27.71)    # AT4
            ]
            
            vlti = cls(pupil=pupil_at, size=diameter, positions=positions,
                      name="VLTI-ATs", latitude=-24.627*u.deg, 
                      longitude=-70.404*u.deg, altitude=2635*u.m)
        
        return vlti
    
    @classmethod
    def life(cls) -> 'TelescopeArray':
        """Create a LIFE (Large Interferometer For Exoplanets) configuration.
        
        LIFE is a proposed space-based nulling interferometer mission concept 
        for direct detection and characterization of exoplanets. It consists 
        of 4 free-flying telescope spacecraft arranged in a planar formation.
        
        Returns
        -------
        life : TelescopeArray
            LIFE interferometric array with 4 telescopes in space.
        
        Notes
        -----
        Since LIFE operates in space, we model it as being at the North Pole 
        (latitude=90°) looking vertically upward. This configuration ensures:
        - Perfect rotation of the array as Earth rotates
        - No atmospheric turbulence
        - Continuous observation geometry
        
        The baseline configuration is based on the LIFE mission concept with:
        - 4 telescopes of 2m diameter each
        - Baselines: 100m to 608m (rectangular configuration)
        - **Centered array**: all telescopes orbit around the central point (0,0)
        - All telescopes are equidistant (~304m) from the array center
        - Planar arrangement in the XY plane
        
        Reference: PHISE project (https://github.com/VForiel/PHISE)
        
        Examples
        --------
        >>> # Create LIFE array
        >>> life = TelescopeArray.life()
        >>> print(f"LIFE: {life.num_telescopes} telescopes")
        >>> life.plot_array(show_pupils=True)
        
        >>> # Check it's interferometric
        >>> print(f"Interferometric: {life.is_interferometric()}")
        """
        # Simple circular pupil for LIFE telescopes (2m diameter)
        pupil_life = Pupil(diameter=2.0*u.m)
        pupil_life.add_disk(radius=1.0*u.m)
        diameter = 2.0 * u.m
        
        # LIFE baseline configuration (from PHISE get_LIFE_telescopes)
        # Centered configuration: all telescopes orbit around central point (0,0)
        # Original PHISE positions centered to ensure (0,0) is the array center
        positions_original = [
            (0, 0),        # Telescope 1
            (100, 0),      # Telescope 2
            (0, 600),      # Telescope 3
            (100, 600)     # Telescope 4
        ]
        
        # Center the array: compute centroid and shift all positions
        centroid_x = sum(p[0] for p in positions_original) / len(positions_original)
        centroid_y = sum(p[1] for p in positions_original) / len(positions_original)
        
        positions = [
            (x - centroid_x, y - centroid_y) for x, y in positions_original
        ]
        
        # Space-based: North Pole configuration for perfect Earth rotation tracking
        life = cls(pupil=pupil_life, size=diameter, positions=positions,
                  name="LIFE", latitude=90*u.deg, longitude=0*u.deg, 
                  altitude=0*u.m)  # altitude=0 for space (not applicable)
        
        return life
    
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
        
        # Handle empty positions list gracefully
        if len(self.positions) == 0:
            ax.text(0.5, 0.5, "No Telescopes Defined", ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{self.name} - Empty Configuration')
            return ax

        baselines = self.get_baseline_array()
        
        if show_pupils:
            # Determine bounding box of all telescope positions
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            
            size = self.size.to(u.m).value if hasattr(self.size, 'to') else float(self.size)
            radius = size / 2.0
            
            for x, y in self.positions:
                min_x = min(min_x, x - radius)
                max_x = max(max_x, x + radius)
                min_y = min(min_y, y - radius)
                max_y = max(max_y, y + radius)
                
            # If no positions or single point, handle defaults
            if min_x == float('inf'):
                cx, cy = 0.0, 0.0
                span = 10.0 # Default span
            else:
                cx = (min_x + max_x) / 2.0
                cy = (min_y + max_y) / 2.0
                span_x = max_x - min_x
                span_y = max_y - min_y
                span = max(span_x, span_y)
                if span == 0: span = size * 2.0 # Fallback for single point
            
            # Add margin
            margin = span * 0.15
            canvas_span = span + 2 * margin
            
            # Setup canvas
            npix_canvas = int(canvas_span * 10) # 10 pixels/meter resolution base
            npix_canvas = max(512, min(npix_canvas, 2048)) # Clamp resolution
            
            # Pixel scale
            pixel_scale = canvas_span / npix_canvas
            
            canvas = np.zeros((npix_canvas, npix_canvas), dtype=float)
            
            # Canvas extent
            extent = [cx - canvas_span/2, cx + canvas_span/2, cy - canvas_span/2, cy + canvas_span/2]
            
            # Render pupil at each position
            diam_m = self.pupil.diameter.to(u.m).value if isinstance(self.pupil.diameter, u.Quantity) else float(self.pupil.diameter)
            diam = diam_m * pupil_scale 
            npix_pupil = int(diam / pixel_scale)
            npix_pupil = max(32, min(npix_pupil, 256))
            pupil_arr = self.pupil.get_array(npix=npix_pupil, soft=True)
            
            for x_pos, y_pos in self.positions:
                # Calculate pixel position on canvas relative to bottom-left corner of extent
                x_pix = int((x_pos - extent[0]) / pixel_scale)
                y_pix = int((y_pos - extent[2]) / pixel_scale)
                
                half_npix = npix_pupil // 2
                
                # Insert pupil
                x_start_can = max(0, x_pix - half_npix)
                x_end_can = min(npix_canvas, x_pix + half_npix)
                y_start_can = max(0, y_pix - half_npix)
                y_end_can = min(npix_canvas, y_pix + half_npix)
                
                # Slices in pupil array
                # If canvas start > intended start, we cropped left/bottom side of pupil
                x_crop_start = max(0, -(x_pix - half_npix)) 
                y_crop_start = max(0, -(y_pix - half_npix))
                
                # Length to copy
                len_x = x_end_can - x_start_can
                len_y = y_end_can - y_start_can
                
                px_start = x_crop_start
                px_end = px_start + len_x
                py_start = y_crop_start
                py_end = py_start + len_y
                
                if len_x > 0 and len_y > 0:
                    canvas[y_start_can:y_end_can, x_start_can:x_end_can] = np.maximum(
                        canvas[y_start_can:y_end_can, x_start_can:x_end_can],
                        pupil_arr[py_start:py_end, px_start:px_end]
                    )

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
        ax.set_title(f'{self.name} - {mode} ({len(self.positions)} telescope{"s" if len(self.positions) > 1 else ""})')
        
        return ax
    
    def process(self, wavefront: Any, pipeline: Optional['Pipeline'] = None) -> Any:
        """Apply telescope array aperture mask to wavefront.
        
        Takes a single input wavefront and produces N output wavefronts (one per telescope position),
        each with the shared pupil mask applied and position-dependent phase shifts.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to process.
        pipeline : Pipeline, optional
            Simulation pipeline.
        
        Returns
        -------
        wavefront : WavefrontArray or Wavefront
            If single telescope: returns single Wavefront
            If multiple telescopes: returns WavefrontArray with one wavefront per position
        """
        # Create temporary Telescope objects for each position to reuse existing logic
        # This maintains compatibility while using the new architecture
        telescopes = []
        for i, pos in enumerate(self.positions):
            tel = Telescope(pupil=self.pupil, position=pos, size=self.size, 
                          name=f"{self.name}-{i+1}")
            telescopes.append(tel)
        
        # Broadcast single wavefront to all telescope positions
        if wavefront is None:
            raise ValueError("Wavefront input required for TelescopeArray.process()")
        
        wf_list = [wavefront.copy() for _ in range(len(telescopes))]
        
        output_list = []
        locations = []
        for i, tel in enumerate(telescopes):
            wf = wf_list[i]
            # Ensure 3D field shape for downstream components even with single sample
            if wf.ndim == 2:
                wf = wf[np.newaxis, ...]

            # Apply geometric phase (piston + tilt) for off-axis sources
            try:
                wavelength = wf.wavelength if hasattr(wf, 'wavelength') else None
            except Exception:
                wavelength = None
            if wavelength is not None and hasattr(wf, 'source_directions') and wf.source_directions is not None:
                cx, cy = tel.position
                k = 2 * np.pi / wavelength.to(u.m).value
                # Build local coordinate grid from wavefront size
                try:
                    size_m = wf.width.to(u.m).value
                except Exception:
                    size_m = float(wf.width)
                npix = wf.npix
                u_vec = np.linspace(-size_m/2, size_m/2, npix)
                v_vec = np.linspace(-size_m/2, size_m/2, npix)
                U, V = np.meshgrid(u_vec, v_vec)
                dirs = wf.source_directions
                for s in range(wf.shape[0]):
                    tx = u.Quantity(dirs[s][0], u.rad).to(u.rad).value
                    ty = u.Quantity(dirs[s][1], u.rad).to(u.rad).value
                    piston = k * (cx * tx + cy * ty)
                    tilt = k * (U * tx + V * ty)
                    phasor = np.exp(1j * (piston + tilt))
                    wf[s] *= phasor
            
            # Apply telescope pupil
            wf_processed = tel.process(wf)
            output_list.append(wf_processed)
            locations.append(tel.position)
        
        # If single telescope, return single Wavefront for backward compatibility
        if len(output_list) == 1:
            return output_list[0]
        return WavefrontArray(output_list, locations=locations)


# Backward compatibility aliases
Collector = Telescope  # Old name for Telescope class
# Note: Old TelescopeArray API is not directly compatible due to architectural changes
# Users should migrate to new constructor: TelescopeArray(pupil=..., size=..., positions=...)
