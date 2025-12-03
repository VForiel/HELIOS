
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from copy import deepcopy as copy

# Ensure we can import helios from src
sys.path.insert(0, os.path.abspath('./src'))

import helios
import helios.components.photonics as photonics
import helios.components.fibers as fibers

def generate_mixed_uml():
    print("Generating Mixed PathPermutator UML...")

    ctx = helios.Context()

    # 4 inputs
    fiber_in = fibers.FiberIn(modes=1, name="In")
    ctx.add_layer([copy(fiber_in) for _ in range(4)])

    # Mixed Layer: [None, Perm(2), None]
    # Input 0 -> None -> Output 0
    # Input 1, 2 -> Perm -> Output 2, 1
    # Input 3 -> None -> Output 3
    
    # Note: Swap(mapping=[1, 0]) swaps its 2 inputs.
    perm = photonics.Swap(mapping=[1, 0], name="SwapCenter")
    
    # We need to be careful about inputs.
    # FiberIn produces 1 output each. Total 4 outputs.
    # Layer 2 expects:
    # Branch 0: None (1 input)
    # Branch 1: Perm (2 inputs)
    # Branch 2: None (1 input)
    # Total inputs needed: 1 + 2 + 1 = 4. Matches.
    
    ctx.add_layer([None, perm, None])

    # Output layer
    fiber_out = fibers.FiberOut(name="Out")
    ctx.add_layer([copy(fiber_out) for _ in range(4)])

    # Generate Diagram
    try:
        fig = ctx.plot_uml_diagram(figsize=(12, 8), layer_spacing=2.0)
        plt.savefig('uml_mixed.png')
        print("Diagram saved to uml_mixed.png")
    except Exception as e:
        print(f"Error generating diagram: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_mixed_uml()
