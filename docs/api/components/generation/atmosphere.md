# Atmosphere

```{eval-rst}
.. autoclass:: helios.components.Atmosphere
   :members:
   :undoc-members:
   :show-inheritance:
```

## Atmospheric Physics

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
