
import matplotlib.pyplot as plt
from  helios.components.collector import TelescopeArray
from helios.components.pupil import Pupil
import numpy as np
import astropy.units as u

def verify_centering():
    # Create array far from origin
    # Center at (100, 100)
    tel = TelescopeArray(name="Off-Center Array")
    pupil = Pupil(diameter=1.0*u.m) # Small pupil
    pupil.add_disk(radius=0.5*u.m)
    
    # Add collectors around (100, 100)
    # Positions: (90, 100) and (110, 100) -> Center x=100
    tel.add_collector(pupil=pupil, position=(90, 100), size=1.0*u.m, name="C1")
    tel.add_collector(pupil=pupil, position=(110, 100), size=1.0*u.m, name="C2")
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    tel.plot_array(ax=ax)
    
    # Check limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    print(f"X Limits: {xlim}")
    print(f"Y Limits: {ylim}")
    
    cx = (xlim[0] + xlim[1]) / 2.0
    cy = (ylim[0] + ylim[1]) / 2.0
    
    print(f"Plot Center: ({cx:.1f}, {cy:.1f})")
    
    # Expected center is (100, 100)
    if 99.0 < cx < 101.0 and 99.0 < cy < 101.0:
        print("SUCCESS: Plot is centered on (100, 100).")
    else:
        print("FAILURE: Plot is NOT centered on (100, 100).")
        
    plt.savefig("verify_centering_output.png")
    print("Saved to verify_centering_output.png")

if __name__ == "__main__":
    verify_centering()
