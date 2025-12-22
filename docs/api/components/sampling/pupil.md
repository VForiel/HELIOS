# Pupil

Aperture geometry and transmission definitions.

```{eval-rst}
.. autoclass:: helios.components.Pupil
   :members:
   :undoc-members:
   :show-inheritance:
```

## Pupil Construction

The `Pupil` class builds aperture masks using geometric primitives:

- **Coordinate system**: Pupil diameter in meters, elements positioned relative to center
- **Segmented primaries**: Use `add_segmented_primary(seg_flat, rings, gap)` with flat-to-flat segment size
- **Anti-aliasing**: Use `get_array(npix, soft=True, oversample=4)` for smooth edges
- **Telescope presets**: `Pupil.like('JWST')`, `Pupil.like('VLT')`, `Pupil.like('ELT')`
