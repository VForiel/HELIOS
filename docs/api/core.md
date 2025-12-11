# Core Framework

The core framework provides the fundamental abstractions for building simulations.

## Pipeline and Layers

The simulation pipeline is built using a layered architecture where each component implements the `Layer` interface and processes signals sequentially.

```{eval-rst}
.. automodule:: helios.core.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
```

## Wavefront and Simulation

Physical wavefront representation and simulation utilities.

```{eval-rst}
.. automodule:: helios.core.simulation
   :members:
   :undoc-members:
   :show-inheritance:
```

## Architecture Overview

The core abstraction is a **Layer** pipeline orchestrated by **Pipeline**:

- All components inherit from `Layer` and implement `process(wavefront, pipeline)`
- `Pipeline.observe()` sequentially processes layers: Scene → TelescopeArray → Optics → Detectors
- Layers can be parallel (list of layers) or sequential (single layer)
- Signal flow: `Scene` generates initial `Wavefront`, each layer transforms it, final layer produces output
