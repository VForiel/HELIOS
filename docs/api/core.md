# Core Framework

The core framework provides the fundamental abstractions for building simulations.

## Context and Layers

The simulation pipeline is built using a layered architecture where each component implements the `Layer` interface and processes signals sequentially.

```{eval-rst}
.. automodule:: helios.core.context
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

The core abstraction is a **Layer** pipeline orchestrated by **Context**:

- All components inherit from `Layer` and implement `process(wavefront, context)`
- `Context.observe()` sequentially processes layers: Scene → Collectors → Optics → Detectors
- Layers can be parallel (list of layers) or sequential (single layer)
- Signal flow: `Scene` generates initial `Wavefront`, each layer transforms it, final layer produces output
