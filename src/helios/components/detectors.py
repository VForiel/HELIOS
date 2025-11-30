import numpy as np
from typing import Tuple, Optional
from astropy import units as u
from ..core.context import Element, Context
from ..core.simulation import Wavefront

class Camera(Element):
    """
    Detector camera.
    
    Parameters
    ----------
    pixels : tuple of int, optional
        Number of pixels (width, height). Default: (1024, 1024)
    dark_current : astropy.Quantity, optional
        Dark current rate. Default: 0 e-/s
    integration_time : astropy.Quantity, optional
        Integration time. Default: 1 s
    name : str, optional
        Name of the camera for identification in diagrams
    """
    def __init__(self, pixels: Tuple[int, int] = (1024, 1024), 
                 dark_current: u.Quantity = 0*u.electron/u.s, 
                 integration_time: u.Quantity = 1*u.s, 
                 name: Optional[str] = None, **kwargs):
        super().__init__(name=name or "Camera")
        self.pixels = pixels
        self.dark_current = dark_current
        self.integration_time = integration_time

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
