# Scene Components

Celestial objects and scene composition for astronomical simulations.

## Overview

Scene components define the astrophysical sources that generate light in the simulation. All objects inherit from `CelestialBody` and provide spectral energy distributions (SEDs) based on physical parameters.

## Scene Container

```{eval-rst}
.. autoclass:: helios.components.Scene
   :members:
   :undoc-members:
   :show-inheritance:
```

## Celestial Bodies

### Base Class

```{eval-rst}
.. autoclass:: helios.components.scene.CelestialBody
   :members:
   :undoc-members:
   :show-inheritance:
```

### Star

```{eval-rst}
.. autoclass:: helios.components.Star
   :members:
   :undoc-members:
   :show-inheritance:
```

### Planet

```{eval-rst}
.. autoclass:: helios.components.Planet
   :members:
   :undoc-members:
   :show-inheritance:
```

### ExoZodiacal Dust

```{eval-rst}
.. autoclass:: helios.components.ExoZodiacal
   :members:
   :undoc-members:
   :show-inheritance:
```

### Zodiacal Light

```{eval-rst}
.. autoclass:: helios.components.Zodiacal
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utility Functions

```{eval-rst}
.. autofunction:: helios.components.scene.modified_blackbody
```

## Physical Models

### Spectral Energy Distributions

All celestial bodies use modified blackbody radiation with optional dust emission parameter β:

$$
B_\lambda(T, \beta) = B_\lambda^{BB}(T) \cdot \lambda^{-\beta}
$$

where:
- $B_\lambda^{BB}(T)$ is the Planck blackbody function
- $T$ is the effective temperature
- $\beta$ is the emissivity index (β=0 for pure blackbody, β>0 for dust-like emission)

### Default Temperatures

- **Star**: 5778 K (solar temperature)
- **Planet**: 300 K (warm Jupiter)
- **ExoZodiacal**: 270 K (warm dust)
- **Zodiacal**: 5700 K (solar-illuminated dust, forward-scattering dominated)
