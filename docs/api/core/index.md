# Core Framework

The core framework provides the fundamental abstractions for building simulations.

```{toctree}
:maxdepth: 2

wavefront
pipeline
layer
component
```

## Architecture Overview

The core abstraction is a **Layer** pipeline orchestrated by **Pipeline**:

- All components inherit from `Layer` and implement `process(wavefront, pipeline)`
- `Pipeline.observe()` sequentially processes layers: Scene → TelescopeArray → Optics → Detectors
- Layers can be parallel (list of layers) or sequential (single layer)
- Signal flow: `Scene` generates initial `Wavefront`, each layer transforms it, final layer produces output