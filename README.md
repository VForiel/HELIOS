# HELIOS
**Hierarchical End-to-end Lightpath & Instrumental response Simulation**

HELIOS is a Python module designed for high-performance end-to-end simulation of astronomical observations. It features a layered architecture allowing flexible composition of scenes, optics, and detectors.

## Features
- **Layered Architecture**: Easily stack components like scenes, telescopes, coronagraphs, and detectors.
- **Physical Units**: Built-in support for `astropy.units`.
- **Performance**: C++ optimized extensions for computationally intensive tasks.
- **Extensible**: Modular design allows easy addition of new components.

## Installation
```bash
pip install .
```

## Usage Example
```python
import helios
from astropy import units as u

# Create a scene
scene = helios.components.Scene()
scene.add(helios.components.Star(distance=10*u.pc, temperature=5700*u.K, magnitude=5))

# Define collectors
collectors = helios.components.Collectors(latitude=0*u.deg, longitude=0*u.deg, altitude=2000*u.m)
pupil = helios.components.Pupil(segments=1)
collectors.add(size=8*u.m, shape=pupil, position=(0,0))

# Setup context
context = helios.Context()
context.add_layer(scene)
context.add_layer(collectors)
context.add_layer(helios.components.Camera(pixels=(1024, 1024)))

# Run simulation
image = context.observe()
```

## Documentation
Full documentation is available on ReadTheDocs.
