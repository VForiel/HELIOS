"""
HELIOS: Hierarchical End-to-end Lightpath & Instrumental response Simulation
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
from .core.context import Context, Layer
from .core.simulation import Simulation, Wavefront

# Expose submodules
from . import components
from . import core

# Expose all component classes directly at package level for convenience
from .components import (
    # Scene components
    Scene, Star, Planet, ExoZodiacal, Zodiacal,
    # Optical components
    Collectors, Interferometer, Pupil, Coronagraph, BeamSplitter, FiberIn, FiberOut, Atmosphere, AdaptiveOptics,
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
    'Context', 'Layer', 'Simulation', 'Wavefront',
    # Submodules
    'components', 'core',
    # Scene components
    'Scene', 'Star', 'Planet', 'ExoZodiacal', 'Zodiacal',
    # Optical components
    'Collectors', 'Interferometer', 'Pupil', 'Coronagraph', 'BeamSplitter', 'FiberIn', 'FiberOut', 'Atmosphere', 'AdaptiveOptics',
    # Detector components
    'Camera',
    # Photonic components
    'PhotonicChip', 'TOPS', 'MMI',
]

