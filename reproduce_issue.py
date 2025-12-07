
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import helios
from helios.components.collector import TelescopeArray
from helios.components.pupil import Pupil
import astropy.units as u

try:
    print("reproducing custom array issue...")
    
    # Simulate what app.py does
    telescope = helios.TelescopeArray(name="Custom Array")
    
    # 1. Add a single collector (mimic Custom default)
    col_data = {
        'x': 0.0, 'y': 0.0, 'diameter': 8.0, 
        'pupil_type': 'Circular', 'central_obstruction': 0.0, 'spiders': 0
    }
    
    p = helios.Pupil(diameter=col_data['diameter'] * u.m)
    p.add_disk(radius=col_data['diameter']/2 * u.m)
    
    telescope.add_collector(
        pupil=p, 
        position=(col_data['x'] * u.m, col_data['y'] * u.m), 
        size=col_data['diameter'] * u.m, 
        name="T1"
    )
    
    print(f"Collectors: {len(telescope.collectors)}")
    for c in telescope.collectors:
        print(f"Col: {c.position}, Size: {c.size}")

    # Plot
    print("Attempting to plot...")
    ax = telescope.plot_array()
    print("Plot created successfully.")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
