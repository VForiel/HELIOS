# HELIOS Documentation

**Hierarchical End-to-end Lightpath & Instrumental response Simulation**

HELIOS is a Python framework for end-to-end simulation of astronomical observations, from celestial scenes through optical systems to detector outputs.

## Architecture Overview

HELIOS uses a **layered pipeline architecture** where light propagates sequentially through components, from astronomical sources to detector output:

```{mermaid}
flowchart LR
    subgraph Context["Context (Orchestrator)"]
        observe["observe()"]
    end
    
    subgraph Scene["Scene Layer"]
        direction LR
        Star["⭐ Star"]
        Planet["🪐 Planet"]
        Zodi["Zodiacal"]
        ExoZodi["ExoZodiacal"]
    end
    
    subgraph Atm_Collectors["Atmosphere + Collectors"]
        direction LR
        Atm["🌫️ Atmosphere<br/>(Turbulence)"]
        Collectors["🔭 Collectors<br/>(Telescope Apertures)"]
        Atm --> |Wavefront| Collectors
    end
    
    subgraph Optics["Optical Layers"]
        direction LR
        Pupil["◯ Pupil<br/>(Aperture Geometry)"]
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
    
    Scene --> |Wavefront| Atm_Collectors
    Atm_Collectors --> |Wavefront| Optics
    Optics --> BS
    BS --> |Split Beams| Photonics
    BS --> |Direct Path| Camera
    Photonics --> |Coupled Light| Camera
    Camera --> |ndarray| Result["Final Image"]
    
    Context -.-> |orchestrates| Scene
    Context -.-> |orchestrates| Atm_Collectors
    Context -.-> |orchestrates| Optics
    Context -.-> |orchestrates| Photonics
    Context -.-> |orchestrates| Detectors
    
    style Scene fill:#fff4e6
    style Atm_Collectors fill:#e1f5fe
    style Optics fill:#e3f2fd
    style Photonics fill:#f3e5f5
    style Detectors fill:#e8f5e9
    style Context fill:#fce4ec
    style Result fill:#ffeb3b
```

**Key Concepts:**
- **Sequential Processing**: Solid arrows show the wavefront propagation path
- **Parallel Processing**: Multiple branches after BeamSplitter
- **Context Orchestration**: Dotted lines show the Context managing all layers
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
