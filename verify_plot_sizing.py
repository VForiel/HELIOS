
import matplotlib.pyplot as plt
from  helios.components.collector import TelescopeArray
from helios.components.pupil import Pupil
import numpy as np
import astropy.units as u

def verify_sizing():
    # Create single telescope
    tel = TelescopeArray(name="Test Telescopye")
    pupil = Pupil(diameter=8.0*u.m)
    pupil.add_disk(radius=4.0*u.m)
    tel.add_collector(pupil=pupil, position=(0,0), size=8.0*u.m)
    
    # Plot
    ax = tel.plot_array()
    
    # Get extent
    # Ax limits should be roughly -4.6m to +4.6m (4m radius + 15% margin = 4.6m)
    # Total span should be ~9.2m
    xlim = ax.get_xlim()
    span = xlim[1] - xlim[0]
    
    print(f"X Limits: {xlim}")
    print(f"Span: {span} m")
    
    # Expected span: Diameter (8m) / 2 = Radius (4m). 
    # Extent = Radius + 15% = 4.6m.
    # Total Span = 9.2m.
    
    # Previous incorrect span was: Diameter (8m) + 15% = 9.2m extent -> Total Span ~18.4m
    
    if 8.0 < span < 10.0:
        print("SUCCESS: Span is reasonable (~9.2m).")
    elif span > 16.0:
        print("FAILURE: Span is too large (~18m), fix not working.")
    else:
        print(f"WARNING: Unexpected span {span}")
        
    plt.savefig("verify_sizing_output.png")
    print("Saved verification plot to verify_sizing_output.png")

if __name__ == "__main__":
    verify_sizing()
