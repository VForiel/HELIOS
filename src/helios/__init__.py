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
from .core.context import Context
from .core.simulation import Simulation

# Expose submodules
from . import components
from . import core
