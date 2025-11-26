# Photonic Components

Integrated photonic circuits for interferometry and beam combination.

## Overview

Photonic components model on-chip optical circuits used in astronomical interferometry. These components enable compact, stable beam combination for high-precision measurements.

## Photonic Chip

```{eval-rst}
.. autoclass:: helios.components.PhotonicChip
   :members:
   :undoc-members:
   :show-inheritance:
```

### Photonic Integration

Photonic integrated circuits (PICs) replace bulk optics with waveguide-based components, providing:
- **Stability**: Monolithic fabrication eliminates alignment drift
- **Compactness**: Millimeter-scale footprint vs. meter-scale bulk optics
- **Manufacturability**: Lithographic fabrication enables complex designs

## TOPS (Three-Output Pupil Slicer)

```{eval-rst}
.. autoclass:: helios.components.TOPS
   :members:
   :undoc-members:
   :show-inheritance:
```

### Pupil Slicing Concept

TOPS devices divide the telescope pupil into multiple sub-apertures, each coupled into a separate waveguide. This enables:
- Spatial filtering for single-mode behavior
- Baseline diversity from sub-aperture pairs
- Simultaneous multiple baseline measurements

## MMI (Multi-Mode Interferometer)

```{eval-rst}
.. autoclass:: helios.components.MMI
   :members:
   :undoc-members:
   :show-inheritance:
```

### Self-Imaging Principle

MMI couplers exploit the self-imaging property of multi-mode waveguides to create fixed splitting ratios. Light entering the MMI reproduces itself at periodic distances, enabling:
- Broadband operation (achromatic splitting)
- Low loss (no mode conversion required)
- Compact footprint (typically <1 mm length)

## Applications

Photonic components in HELIOS are designed for:
- **Nulling interferometry**: Destructive combination to suppress starlight
- **Closure phase**: Higher-order observables immune to atmospheric piston
- **Kernel-nulling**: Optimal combination of sub-apertures for exoplanet detection
