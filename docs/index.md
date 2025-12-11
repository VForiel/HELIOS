# HELIOS Documentation

**Hierarchical End-to-end Lightpath & Instrumental response Simulation**

HELIOS is a Python framework for end-to-end simulation of astronomical observations, from celestial scenes through optical systems to detector outputs.

## Architecture Overview

HELIOS uses a **layered pipeline architecture** where light propagates sequentially through components, from astronomical sources to detector output:

```{mermaid}
flowchart LR
    subgraph Pipeline["Pipeline (Orchestrator)"]
        observe["observe()"]
    end
    
    subgraph Scene["Scene Layer"]
        direction LR
        Star["⭐ Star"]
        Planet["🪐 Planet"]
        Zodi["Zodiacal"]
        ExoZodi["ExoZodiacal"]
    end
    
    subgraph TelescopeGroup["Telescope Array"]
        direction TB
        Collector["Collector<br/>(Single Aperture)"]
        TelescopeArray["🔭 TelescopeArray<br/>(Single/Interferometric)"]
        Collector -.-> |"collected by"| TelescopeArray
    end
    
    subgraph Optics["Optical Layers"]
        direction LR
        Pupil["◯ Pupil<br/>(Aperture Geometry)"]
        Atm["🌫️ Atmosphere<br/>(Turbulence)"]
        Coro["✱ Coronagraph<br/>(Starlight Suppression)"]
        AO["🔄 AdaptiveOptics<br/>(Wavefront Correction)"]
        BS["⚡ BeamSplitter"]
    end
    
    subgraph Photonics["Photonic Layers"]
        direction TB
        Chip["💎 PhotonicChip"]
        TOPS["TOPS"]
        MMI["MMI"]
    end
    
    subgraph Detectors["Detector Layers"]
        Camera["📷 Camera<br/>(Terminal Layer)"]
    end
    
    Scene --> |Wavefront| TelescopeGroup
    TelescopeGroup --> |"Wavefront<br/>(single/interferometric)"| Optics
    Optics --> BS
    BS --> |Split Beams| Photonics
    BS --> |Direct Path| Camera
    Photonics --> |Coupled Light| Camera
    Camera --> |ndarray| Result["Final Image"]
    
    Pipeline -.-> |orchestrates| Scene
    Pipeline -.-> |orchestrates| TelescopeGroup
    Pipeline -.-> |orchestrates| Optics
    Pipeline -.-> |orchestrates| Photonics
    Pipeline -.-> |orchestrates| Detectors
    
    style Scene fill:#fff4e6
    style TelescopeGroup fill:#e1f5fe
    style Optics fill:#e3f2fd
    style Photonics fill:#f3e5f5
    style Detectors fill:#e8f5e9
    style Pipeline fill:#fce4ec
    style Result fill:#ffeb3b
```

**Key Concepts:**
- **Sequential Processing**: Solid arrows show the wavefront propagation path
- **Parallel Processing**: Multiple branches after BeamSplitter
- **Pipeline Orchestration**: Dotted lines show the Pipeline managing all layers
- **Terminal Layer**: Camera produces final numpy array output

## Features

- **Layered Architecture**: Flexible composition of scenes, optics, and detectors
- **Physical Units**: Built-in support for `astropy.units`
- **Scientific Rigor**: Physics-based simulations with validation
- **Educational Clarity**: Scientifically rigorous but explained for all scientists
- **AI-First Development**: Maintained by AI agents following strict quality standards

```{toctree}
:maxdepth: 2
:caption: Contents:

api/index
contribute
architecture
```

## Quick Start

```python
import helios
from astropy import units as u

# Create a scene
scene = helios.Scene(distance=10*u.pc)
scene.add(helios.Star(temperature=5700*u.K, magnitude=5))

# Define telescope array (automatically detects single vs interferometric mode)
telescope = helios.TelescopeArray(latitude=0*u.deg, longitude=0*u.deg)
pupil = helios.Pupil(diameter=8*u.m)
telescope.add_collector(pupil=pupil, position=(0, 0), size=8*u.m)

# For interferometry, add more collectors at different positions:
# telescope.add_collector(pupil=pupil, position=(47, 0), size=8*u.m)
# telescope.is_interferometric()  # Returns True if multiple baselines

# Setup pipeline and run
pipeline = helios.Pipeline()
pipeline.add_layer(scene)
pipeline.add_layer(telescope)
pipeline.add_layer(helios.Camera(pixels=(1024, 1024)))

image = pipeline.observe()
```
