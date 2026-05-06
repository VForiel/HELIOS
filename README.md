# HELIOS
**Hierarchical End-to-end Lightpath & Instrumental response Observational Simulation**

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

pupil = helios.Pupil(diameter=8*u.m)

# Define telescope array (automatic single/interferometric detection)
telescope = helios.TelescopeArray(
    pupil=pupil,
    size=8*u.m,
    positions=[(0, 0)],
    latitude=0*u.deg,
    longitude=0*u.deg,
    altitude=2000*u.m,
)

# For interferometry, add more positions at different baselines:
# telescope.add_position(47, 0)
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

## Web Interface
A web-based graphical interface is available to easily experiment with HELIOS.
See `web/README.md` for instructions on how to run it using Docker.

