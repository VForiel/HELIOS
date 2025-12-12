# 🚀 Getting Started

Welcome to HELIOS! This guide will help you get started with simulating astronomical observations.

## Installation

### Prerequisites

- Python 3.10 or higher (Python 3.13.9 recommended)
- pip package manager

### Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/VForiel/HELIOS.git
cd HELIOS
pip install .
```

For development, you may want to install in editable mode:

```bash
pip install -e .
```

## Quick Start

Here's a simple example to get you started with HELIOS:

```python
import helios
from astropy import units as u

# Create a scene with a star
scene = helios.Scene(distance=10*u.pc)
scene.add(helios.Star(temperature=5700*u.K, magnitude=5))

# Define a telescope
telescope = helios.TelescopeArray(latitude=0*u.deg, longitude=0*u.deg)
pupil = helios.Pupil(diameter=8*u.m)
telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)

# Setup the observation pipeline
pipeline = helios.Pipeline()
pipeline.add_layer(scene)
pipeline.add_layer(telescope)
pipeline.add_layer(helios.Camera(pixels=(1024, 1024)))

# Run the simulation
image = pipeline.observe()
```

## Basic Concepts

### Layered Architecture

HELIOS uses a **layered pipeline architecture** where light propagates sequentially through components:

1. **Scene Layer** 🌟 - Defines celestial objects (stars, planets, zodiacal light)
2. **Telescope Array** 🔭 - Collects light through apertures (single or interferometric)
3. **Optical Layers** - Processes light (atmosphere, coronagraphs, adaptive optics)
4. **Photonic Layers** 💎 - Integrated photonic circuits (optional)
5. **Detector Layer** 📷 - Converts light to digital images

### Physical Units

HELIOS uses `astropy.units` for all physical quantities:

```python
from astropy import units as u

# Distances
distance = 10 * u.pc  # parsecs
diameter = 8 * u.m    # meters

# Temperatures
temp = 5700 * u.K     # Kelvin

# Wavelengths
wavelength = 550 * u.nm  # nanometers
```

### Single vs Interferometric Mode

The `TelescopeArray` automatically detects whether you're doing single-aperture or interferometric observations:

```python
# Single aperture (one collector)
telescope = helios.TelescopeArray(latitude=0*u.deg, longitude=0*u.deg)
telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)

# Interferometric (multiple collectors at different positions)
telescope.add_collector(pupil=pupil, position=(47, 0), size=8*u.m)
telescope.add_collector(pupil=pupil, position=(0, 47), size=8*u.m)

# Check mode
if telescope.is_interferometric():
    print("Running in interferometric mode")
```

## Next Steps

- 📚 Explore the {doc}`api/index` for detailed component documentation
- ⚙️ Learn about the {doc}`architecture` and design principles
- 🤝 Read the {doc}`contribute` guide if you want to contribute

## Examples

Check out the `examples/` directory in the repository for more complete demonstrations:

- Basic imaging simulations
- Coronagraphic observations
- Interferometric setups
- Photonic integrated circuits

## Getting Help

If you encounter issues or have questions:

1. Check the {doc}`api/index` for detailed documentation
2. Review the {doc}`architecture` page for design concepts
3. Open an issue on [GitHub](https://github.com/VForiel/HELIOS/issues)
