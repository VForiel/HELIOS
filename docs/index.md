# HELIOS Documentation

**Hierarchical End-to-end Lightpath & Instrumental response Simulation**

HELIOS is a Python framework for end-to-end simulation of astronomical observations, from celestial scenes through optical systems to detector outputs.

## Features

- **Layered Architecture**: Flexible composition of scenes, optics, and detectors
- **Physical Units**: Built-in support for `astropy.units`
- **Scientific Rigor**: Physics-based simulations with validation
- **AI-First Development**: Maintained by AI agents following strict quality standards

```{toctree}
:maxdepth: 2
:caption: Contents:

contribute
```

## Quick Start

```python
import helios
from astropy import units as u

# Create a scene
scene = helios.components.Scene(distance=10*u.pc)
scene.add(helios.components.Star(temperature=5700*u.K, magnitude=5))

# Define collectors
collectors = helios.components.Collectors(latitude=0*u.deg, longitude=0*u.deg)
collectors.add(size=8*u.m, shape=helios.components.Pupil())

# Setup context and run
context = helios.Context()
context.add_layer(scene)
context.add_layer(collectors)
context.add_layer(helios.components.Camera(pixels=(1024, 1024)))

image = context.observe()
```
