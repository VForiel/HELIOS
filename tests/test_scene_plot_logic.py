import sys
import os
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from helios.components.scene import Scene, Star, Planet

def test_plot_logic():
    print("Testing Scene.plot logic...")
    
    # Create scene
    scene = Scene(distance=10*u.pc)
    
    # 1. Star at non-zero position (should be forced to 0,0)
    star = Star(position=(10*u.AU, 10*u.AU))
    scene.add(star)
    
    # 2. Planet with distance units (1 AU at 10 pc is 0.1 arcsec)
    # 1 AU / 10 pc = 1 / 2062650 * 206265 = 0.1 arcsec
    planet_dist = Planet(mass=1*u.M_jup, position=(1*u.AU, 0*u.AU))
    scene.add(planet_dist)
    
    # 3. Planet with angular units
    planet_angle = Planet(mass=1*u.M_jup, position=(0*u.arcsec, 0.5*u.arcsec))
    scene.add(planet_angle)
    
    # Plot
    fig, ax = scene.plot()
    
    # Verify positions
    # We need to access the collections in the plot
    # There should be 3 points plotted.
    # However, ax.scatter might return a PathCollection.
    # Let's inspect the data in the collections.
    
    collections = ax.collections
    # We iterate through collections to find our points. 
    # Since we added objects sequentially, they might be in the same collection or different ones depending on how scatter is called.
    # In the code, scatter is called inside the loop, so there should be one collection per object.
    
    assert len(collections) == 3, f"Expected 3 collections, got {len(collections)}"
    
    # Star (first object)
    star_offsets = collections[0].get_offsets()
    print(f"Star offsets: {star_offsets}")
    assert np.allclose(star_offsets, [[0, 0]]), "Star should be at (0,0)"
    
    # Planet Dist (second object)
    planet_dist_offsets = collections[1].get_offsets()
    print(f"Planet (dist) offsets: {planet_dist_offsets}")
    # Expected: 1 AU / 10 pc in arcsec.
    expected_x = (1 * u.AU / (10 * u.pc)).to(u.arcsec, equivalencies=u.dimensionless_angles()).value
    assert np.allclose(planet_dist_offsets, [[expected_x, 0]]), f"Planet (dist) should be at ({expected_x}, 0)"
    
    # Planet Angle (third object)
    planet_angle_offsets = collections[2].get_offsets()
    print(f"Planet (angle) offsets: {planet_angle_offsets}")
    assert np.allclose(planet_angle_offsets, [[0, 0.5]]), "Planet (angle) should be at (0, 0.5)"
    
    print("All plot logic tests passed!")
    plt.close(fig)

if __name__ == "__main__":
    test_plot_logic()
