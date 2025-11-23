import numpy as np
from astropy import units as u
from typing import Tuple, List, Union
from ..core.context import Layer, Context
from ..core.simulation import Wavefront

class Pupil:
    """
    Represents the pupil shape of a collector.
    """
    def __init__(self, segments: int = 1):
        self.segments = segments
        self.elements = []

    def add(self, element):
        self.elements.append(element)

    @staticmethod
    def spider(arms: int = 3):
        return {"type": "spider", "arms": arms}

    @staticmethod
    def secondary(size: float = 0.1):
        return {"type": "secondary", "size": size}
    
    @staticmethod
    def like(name: str):
        # Placeholder for preset pupils like JWST
        return Pupil()

class Collectors(Layer):
    """
    Represents the light collectors (telescopes).
    """
    def __init__(self, latitude: u.Quantity = 0*u.deg, longitude: u.Quantity = 0*u.deg, altitude: u.Quantity = 0*u.m, **kwargs):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.collectors = []
        super().__init__()

    def add(self, size: u.Quantity, shape: Pupil, position: Tuple[float, float], **kwargs):
        self.collectors.append({
            "size": size,
            "shape": shape,
            "position": position,
            **kwargs
        })

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Apply pupil mask to wavefront
        # Placeholder logic
        return wavefront

class BeamSplitter(Layer):
    def __init__(self, cutoff: float = 0.5):
        self.cutoff = cutoff
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> List[Wavefront]:
        # Split wavefront into two
        return [wavefront, wavefront] # Placeholder for copy

class Coronagraph(Layer):
    def __init__(self, phase_mask: str = '4quadrants'):
        self.phase_mask = phase_mask
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Apply coronagraphic mask
        return wavefront

class FiberIn(Layer):
    def __init__(self, modes: int = 1, **kwargs):
        self.modes = modes
        super().__init__()
    
    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Couple light into fiber
        return wavefront

class FiberOut(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Light exiting fiber
        return wavefront

def test_collectors():
    c = Collectors(latitude=0*u.deg, longitude=0*u.deg, altitude=0*u.m)
    p = Pupil(segments=1)
    c.add(size=8*u.m, shape=p, position=(0,0))
    assert len(c.collectors) == 1

    # Test defaults
    default_c = Collectors()
    assert default_c.latitude == 0*u.deg
    assert default_c.longitude == 0*u.deg
    assert default_c.altitude == 0*u.m

if __name__ == "__main__":
    test_collectors()
    print("Optics tests passed.")
