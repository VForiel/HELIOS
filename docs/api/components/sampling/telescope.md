# Telescope

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