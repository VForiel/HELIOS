import numpy as np
from astropy import units as u

class Wavefront:
    """
    Represents the electromagnetic field (complex amplitude).
    """
    def __init__(self, wavelength: u.Quantity, size: int):
        self.wavelength = wavelength
        self.field = np.ones((size, size), dtype=np.complex128)
        self.pixel_scale = 1.0 * u.m # Placeholder

    def propagate(self, distance: u.Quantity):
        """
        Propagate the wavefront by a certain distance.
        """
        # Placeholder for Fresnel/Fraunhofer propagation
        pass

class Simulation:
    """
    Helper class for running specific simulation types if needed.
    """
    pass

def test_wavefront_init():
    wf = Wavefront(wavelength=600*u.nm, size=128)
    assert wf.field.shape == (128, 128)
    assert wf.wavelength == 600 * u.nm

if __name__ == "__main__":
    test_wavefront_init()
    print("Simulation tests passed.")
