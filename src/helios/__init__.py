"""
HELIOS: Hierarchical End-to-end Lightpath & Instrumental response Observational Simulation
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("helios")
except PackageNotFoundError:
    __version__ = "unknown"

# Expose core components
from .core.pipeline import Pipeline, Context
from .core.layer import Layer, GenerationLayer, SamplingLayer, OpticalLayer, DetectionLayer, DataLayer
from .core.component import Component, GenerationComponent, SamplingComponent, OpticalComponent, DetectionComponent, DataComponent
from .core.wavefront import Wavefront
from .core.optical_scene import OpticalScene, Spectrum

# Expose submodules
from . import components
from . import core
from . import sim

# Expose all component classes directly at package level for convenience
from .components import (
    # Scene components
    PlanetarySystem, Scene, Star, Planet, ExoZodiacal, Zodiacal,
    # Optical components
    Pupil, Collector, TelescopeArray, Telescope, Coronagraph, BeamSplitter, FiberIn, FiberOut, Atmosphere, AdaptiveOptics, Lens,
    # Detector components
    Camera,
    # Photonic components
    PhotonicChip, TOPS, MMI
)

# Define public API
__all__ = [
    # Version
    '__version__',
    # Core
    'Context', 'Layer', 'Component', 'Simulation', 'Wavefront', 'Spectrum', 'OpticalScene',
    # Component types
    'GenerationComponent', 'SamplingComponent', 'OpticalComponent', 'DetectionComponent', 'DataComponent',
    # Submodules
    'components', 'core',
    # Scene components
    'PlanetarySystem', 'Scene', 'Star', 'Planet', 'ExoZodiacal', 'Zodiacal',
    # Optical components
    'Pupil', 'Collector', 'TelescopeArray', 'Telescope', 'Coronagraph', 'BeamSplitter', 'FiberIn', 'FiberOut', 'Atmosphere', 'AdaptiveOptics', 'Lens',
    # Detector components
    'Camera',
    # Photonic components
    'PhotonicChip', 'TOPS', 'MMI',
    # Simulations
    'sim',
]

