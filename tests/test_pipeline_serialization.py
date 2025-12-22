
import numpy as np
import os
import json
from astropy import units as u
from helios.components import Scene, Star, Planet
from helios.components import Atmosphere
from helios.components import TelescopeArray, Pupil
from helios.core.pipeline import Pipeline

def test_serialization():
    print("Testing Pipeline Serialization...")
    
    # 1. Create a full Pipeline
    pipe = Pipeline(date="2025-01-01T00:00:00", declination=-20*u.deg)
    
    # Scene
    scene = Scene(name="Test System", distance=10*u.pc)
    star = Star(name="Sun", temperature=5778*u.K, magnitude=5.0, mass=1.0*u.M_sun)
    planet = Planet(name="Earth", mass=1.0*u.M_earth, radius=1.0*u.R_earth, 
                   position=(1.0*u.au, 0*u.au), temperature=300*u.K, albedo=0.3)
    scene.add(star)
    scene.add(planet)
    pipe.add_layer(scene)
    
    # Atmosphere
    atm = Atmosphere(rms=100*u.nm, wind_speed=(5*u.m/u.s, 2*u.m/u.s), seed=42)
    pipe.add_layer(atm)
    
    # Telescope
    vlti = TelescopeArray.vlti(uts=True) # 4 collectors
    pipe.add_layer(vlti)
    
    # Camera
    cam = Camera(pixels=(256, 256), read_noise=5*u.electron, dark_current=0.1*u.electron/u.s)
    pipe.add_layer(cam)
    
    print("\nOriginal Pipeline created.")
    print(f"Layers: {[type(l).__name__ for l in pipe.layers]}")
    
    # 2. Save to JSON
    filename = "test_pipeline.json"
    pipe.save(filename)
    print(f"\nPipeline saved to {filename}")
    
    # Check file exists
    if os.path.exists(filename):
        print(f"File size: {os.path.getsize(filename)} bytes")
        with open(filename, 'r') as f:
            data = json.load(f)
            # print("JSON keys:", data.keys())
            # print("Layers data:", [l.get('type') for l in data.get('layers', [])])
    
    # 3. Load from JSON
    print("\nLoading Pipeline...")
    loaded_pipe = Pipeline.load(filename)
    print("Pipeline loaded.")
    
    # 4. Verify
    print("\nVerifying Loaded Pipeline...")
    
    # Check Layers
    assert len(loaded_pipe.layers) == len(pipe.layers), f"Layer count mismatch: {len(loaded_pipe.layers)} vs {len(pipe.layers)}"
    
    # Scene
    l_scene = loaded_pipe.layers[0]
    assert isinstance(l_scene, Scene)
    assert l_scene.name == scene.name
    # Check elements (Star, Planet)
    assert len(l_scene.elements) == 2
    l_star = l_scene.elements[0]
    l_planet = l_scene.elements[1]
    assert isinstance(l_star, Star)
    assert isinstance(l_planet, Planet)
    assert l_star.name == "Sun"
    assert l_planet.name == "Earth"
    # Check physical params
    # Note: quantities might have small float diffs or unit conversions, check values
    print(f"Star Mass: {l_star.mass} (Original: {star.mass})")
    print(f"Planet Pos: {l_planet.position} (Original: {planet.position})")
    
    # Atmosphere
    l_atm = loaded_ctx.layers[1]
    assert isinstance(l_atm, Atmosphere)
    print(f"Atm Wind: {l_atm.wind_velocity} (Original: {atm.wind_velocity})")
    assert np.allclose(l_atm.wind_velocity, atm.wind_velocity)
    assert l_atm.rms == atm.rms
    
    # Telescope
    l_tel = loaded_ctx.layers[2]
    assert isinstance(l_tel, TelescopeArray)
    assert len(l_tel.collectors) == 4
    print(f"Telescope config: {l_tel.name}")
    assert l_tel.collectors[0].pupil.diameter == vlti.collectors[0].pupil.diameter
    
    # Camera
    l_cam = loaded_ctx.layers[3]
    assert isinstance(l_cam, Camera)
    assert l_cam.pixels == cam.pixels
    assert l_cam.read_noise == cam.read_noise
    
    print("\nVerification Successful!")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed {filename}")

if __name__ == "__main__":
    try:
        test_serialization()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
