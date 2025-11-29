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
scene = helios.Scene(distance=10*u.pc)
scene.add(helios.Star(temperature=5700*u.K, magnitude=5))

# Define telescope array (automatic single/interferometric detection)
telescope = helios.TelescopeArray(latitude=0*u.deg, longitude=0*u.deg, altitude=2000*u.m)
pupil = helios.Pupil(diameter=8*u.m)
telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)

# For interferometry, add more collectors at different baselines:
# telescope.add_collector(pupil=pupil, position=(47, 0), size=8*u.m)
# telescope.is_interferometric()  # Returns True if multiple non-colocated collectors

# Setup context
context = helios.Context()
context.add_layer(scene)
context.add_layer(telescope)
context.add_layer(helios.Camera(pixels=(1024, 1024)))

# Run simulation
image = context.observe()
```

## Documentation
Full documentation is available on ReadTheDocs.
