# Telescope Array

Interferometric and multi-telescope configurations.

```{eval-rst}
.. autoclass:: helios.components.TelescopeArray
   :members:
   :undoc-members:
   :show-inheritance:
```

## Architecture

**TelescopeArray** manages a collection of telescopes. It automatically detects operation mode:

- **Single telescope**: When all collectors are colocated (same position)
- **Interferometric**: When collectors have different positions (multiple baselines)

Use `is_interferometric()` to check the current mode. The `process()` method automatically adapts its behavior based on the configuration.

**Presets**:
- `TelescopeArray.vlti()`: Very Large Telescope Interferometer
- `TelescopeArray.life()`: LIFE mission concept