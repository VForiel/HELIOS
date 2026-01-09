"""
HELIOS Components Package
========================

This package contains the optical and processing components used to build
simulation pipelines.

The components are organized into subpackages:
- generation: Light source generation (Scene, Atmosphere)
- sampling: Spatial sampling (Pupil, Telescope, TelescopeArray)
- bulk_optics: Bulk optical elements (Lens, BeamSplitter, Coronagraph, AdaptiveOptics)
- photonics: Integrated photonics (Chip, MMI, TOPS, Splitters, Fibers)
- detection: Detectors (Camera)

All components are exposed at the top level for convenience.
"""

# Generation
from .generation.scene import PlanetarySystem, CelestialBody, Star, Planet, ExoZodiacal, Zodiacal

# Backward compatibility alias
Scene = PlanetarySystem
from .generation.atmosphere import Atmosphere

# Sampling
from .sampling.pupil import Pupil
from .sampling.telescope import Telescope
from .sampling.telescope_array import TelescopeArray

# Bulk Optics
from .bulk_optics.lens import Lens
from .bulk_optics.beam_splitter import BeamSplitter
from .bulk_optics.coronagraph import Coronagraph
from .bulk_optics.adaptive_optics import AdaptiveOptics

# Photonics
from .photonics.chip import PhotonicChip
from .photonics.mmi import MultiModeInterferometer, MMI
from .photonics.tops import ThermoOpticPhaseShifter, TOPS
from .photonics.splitter import YSplitter, Swap
from .photonics.fibers import FiberIn, FiberOut

# Detection
from .detection.camera import Camera

# Backwards compatibility alias
Collector = Telescope

__all__ = [
    # Generation
    "Scene", "CelestialBody", "Star", "Planet", "ExoZodiacal", "Zodiacal",
    "Atmosphere",
    
    # Sampling
    "Pupil",
    "Telescope", "TelescopeArray", "Collector",
    
    # Bulk Optics
    "Lens",
    "BeamSplitter",
    "Coronagraph",
    "AdaptiveOptics",
    
    # Photonics
    "PhotonicChip",
    "MultiModeInterferometer", "MMI",
    "ThermoOpticPhaseShifter", "TOPS",
    "YSplitter", "Swap",
    "FiberIn", "FiberOut",
    
    # Detection
    "Camera"
]
