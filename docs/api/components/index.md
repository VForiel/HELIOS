# Components

Simulation components organized by category.

```{toctree}
:maxdepth: 2

scene
optics
detectors
photonics
```

## Component Categories

### Scene Components
Define astronomical objects and their spectral energy distributions.

- {py:class}`~helios.components.Scene` - Container for celestial objects
- {py:class}`~helios.components.Star` - Stellar sources
- {py:class}`~helios.components.Planet` - Planetary companions
- {py:class}`~helios.components.ExoZodiacal` - Exozodiacal dust emission
- {py:class}`~helios.components.Zodiacal` - Zodiacal light background

### Optical Components
Model light collection, propagation, and manipulation.

- {py:class}`~helios.components.Collector` - Single telescope aperture (data object)
- {py:class}`~helios.components.TelescopeArray` - Single or interferometric telescope array
- {py:class}`~helios.components.Pupil` - Aperture geometry definition
- {py:class}`~helios.components.Atmosphere` - Atmospheric turbulence
- {py:class}`~helios.components.Coronagraph` - Starlight suppression
- {py:class}`~helios.components.AdaptiveOptics` - Wavefront correction

### Detector Components
Convert optical signals to detector outputs.

- {py:class}`~helios.components.Camera` - Imaging detector array

### Photonic Components
Integrated photonics for interferometry and beam combination.

- {py:class}`~helios.components.PhotonicChip` - Photonic integrated circuits
- {py:class}`~helios.components.TOPS` - Three-output pupil slicer
- {py:class}`~helios.components.MMI` - Multi-mode interferometer
