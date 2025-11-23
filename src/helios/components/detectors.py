import numpy as np
from typing import Tuple
from astropy import units as u
from ..core.context import Layer, Context
from ..core.simulation import Wavefront

class Camera(Layer):
    def __init__(self, pixels: Tuple[int, int] = (1024, 1024), dark_current: u.Quantity = 0*u.electron/u.s, integration_time: u.Quantity = 1*u.s, **kwargs):
        self.pixels = pixels
        self.dark_current = dark_current
        self.integration_time = integration_time
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> np.ndarray:
        # Detect light, convert to electrons/counts
        # Placeholder: return random noise or simple intensity
        return np.zeros(self.pixels)

def test_camera():
    cam = Camera(pixels=(100, 100))
    assert cam.pixels == (100, 100)

    # Test defaults
    default_cam = Camera()
    assert default_cam.pixels == (1024, 1024)

if __name__ == "__main__":
    test_camera()
    print("Detector tests passed.")
