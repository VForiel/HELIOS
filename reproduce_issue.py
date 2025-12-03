
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import helios
from astropy import units as u
import matplotlib.pyplot as plt

def reproduce():
    print("Setting up HELIOS context...")
    scene = helios.Scene(distance=10*u.pc)
    star = helios.Star(temperature=5700*u.K, magnitude=5)
    scene.add(star)
    
    camera = helios.Camera(pixels=(64, 64))
    
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(camera)
    
    print("Attempting to generate UML diagram as image...")
    try:
        img = ctx.plot_uml_diagram(return_type='image')
        print(f"Success! Image shape: {img.shape}")
    except AttributeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    reproduce()
