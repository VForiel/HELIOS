# Detector Components

Detector arrays that convert optical signals to digital outputs.

## Overview

Detector components are terminal layers in the simulation pipeline - they receive optical signals and produce final output arrays.

## Camera

```{eval-rst}
.. autoclass:: helios.components.Camera
   :members:
   :undoc-members:
   :show-inheritance:
```

### Camera Model

The camera is the most common detector, producing 2D intensity images. It represents the final focal plane where photons are converted to electrons.

**Key Parameters**:
- `pixels`: Detector array dimensions (nx, ny)
- `pixel_size`: Physical pixel size (affects sampling and field of view)
- `quantum_efficiency`: Photon-to-electron conversion efficiency
- `read_noise`: Detector readout noise in electrons
- `dark_current`: Thermal electron generation rate

### Output Format

The camera returns a numpy array with shape matching the `pixels` parameter. Units depend on the simulation configuration but typically represent:
- Photon counts (ideal detector)
- Electron counts (with quantum efficiency)
- ADU (analog-to-digital units, with gain applied)
