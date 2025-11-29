# Optical Components

Light collection, propagation, and manipulation through optical systems.

## Overview

Optical components model how light is collected by telescopes, propagates through the atmosphere, and is manipulated by coronagraphs and adaptive optics systems.

## Pupil Geometry

```{eval-rst}
.. autoclass:: helios.components.Pupil
   :members:
   :undoc-members:
   :show-inheritance:
```

### Pupil Construction

The `Pupil` class builds aperture masks using geometric primitives:

- **Coordinate system**: Pupil diameter in meters, elements positioned relative to center
- **Segmented primaries**: Use `add_segmented_primary(seg_flat, rings, gap)` with flat-to-flat segment size
- **Anti-aliasing**: Use `get_array(npix, soft=True, oversample=4)` for smooth edges
- **Telescope presets**: `Pupil.like('JWST')`, `Pupil.like('VLT')`, `Pupil.like('ELT')`

## Telescope Arrays

```{eval-rst}
.. autoclass:: helios.components.Collector
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: helios.components.TelescopeArray
   :members:
   :undoc-members:
   :show-inheritance:
```

### Architecture

**Collector**: Data object storing a single telescope aperture configuration (pupil, position, size, name).

**TelescopeArray**: Layer class that manages one or more collectors. Automatically detects operation mode:
- **Single telescope**: When all collectors are colocated (same position)
- **Interferometric**: When collectors have different positions (multiple baselines)

Use `is_interferometric()` to check the current mode. The `process()` method automatically adapts its behavior based on the configuration.

**Backward compatibility aliases**: `Telescope` and `Interferometer` both refer to `TelescopeArray`.

## Atmospheric Turbulence

```{eval-rst}
.. autoclass:: helios.components.Atmosphere
   :members:
   :undoc-members:
   :show-inheritance:
```

### Atmospheric Physics

The atmosphere introduces **chromatic** optical path difference (OPD) errors:

$$
\phi(\lambda) = \frac{2\pi \cdot \text{OPD}}{\lambda}
$$

This means shorter wavelengths (blue) experience larger phase aberrations than longer wavelengths (infrared) for the same atmospheric turbulence.

**Temporal Evolution**: Modeled via **frozen-flow turbulence** (Taylor hypothesis) - turbulent screens drift at constant wind velocity.

**Key Parameters**:
- `rms`: OPD RMS in physical units (meters, nanometers) - NOT phase in radians
- `wind_speed`: Wind velocity magnitude or (vx, vy) tuple
- `wind_direction`: Wind direction in degrees (0° = +x axis)
- `seed`: Random seed for reproducible turbulence realizations

## Coronagraphs

```{eval-rst}
.. autoclass:: helios.components.Coronagraph
   :members:
   :undoc-members:
   :show-inheritance:
```

### Coronagraph Types

**Vortex Coronagraph**: Azimuthal phase ramp that diffracts on-axis light outside the geometric pupil.

**4-Quadrant Phase Mask**: Four π phase shifts in quadrants to create destructive interference on-axis.

## Adaptive Optics

```{eval-rst}
.. autoclass:: helios.components.AdaptiveOptics
   :members:
   :undoc-members:
   :show-inheritance:
```

### Zernike Basis

Adaptive optics corrections are represented using Zernike polynomial coefficients. The `coeffs` dictionary maps Zernike mode indices to their amplitudes.

## Beam Manipulation

```{eval-rst}
.. autoclass:: helios.components.BeamSplitter
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: helios.components.FiberIn
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: helios.components.FiberOut
   :members:
   :undoc-members:
   :show-inheritance:
```
