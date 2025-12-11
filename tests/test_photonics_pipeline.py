import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios
from helios.components.photonics import YSplitter, TOPS, MMI
from helios.components.fibers import FiberIn
from astropy import units as u
import numpy as np
import copy

def test_pipeline():
    print("Testing Photonic Pipeline...")
    
    # Setup
    ctx = helios.Context()
    
    # 1. Scene (Placeholder)
    # We'll just manually inject a wavefront for testing without full scene
    wf_in = helios.Wavefront(wavelength=1.55*u.um, size=16)
    
    # 2. FiberIn (1 input -> 1 output)
    fiber_in = FiberIn(modes=1)
    
    # 3. YSplitter (1 input -> 2 outputs)
    ys1 = YSplitter(name="YS1")
    
    # 4. Layer of 2 YSplitters (2 inputs -> 4 outputs)
    ys2_layer = [YSplitter(name="YS2a"), YSplitter(name="YS2b")]
    
    # 5. Layer with MMI and pass-through (4 inputs -> 5 outputs)
    # Input 0 -> Pass-through -> Output 0
    # Input 1, 2 -> MMI 2x3 -> Output 1, 2, 3
    # Input 3 -> Pass-through -> Output 4
    
    # MMI 2x3 matrix (dummy unitary-ish)
    # 2 inputs, 3 outputs. Matrix shape (3, 2)
    mat = np.array([
        [1, 1],
        [1, 0],
        [0, 1]
    ], dtype=complex) / np.sqrt(2) # Not unitary but good for testing routing
    
    mmi = MMI(matrix=mat, name="MMI_2x3")
    
    mixed_layer = [None, mmi, None]
    
    # 6. Cameras (5 inputs -> 5 outputs)
    cameras = [helios.Camera(pixels=(16,16), name=f"Cam{i}") for i in range(5)]
    
    # Build Context
    # We need a source layer to start. Let's make a dummy source layer
    class Source(helios.core.context.Layer):
        def process(self, wf, ctx=None):
            return wf_in
            
    ctx.add_layer(Source(name="Source"))
    ctx.add_layer(fiber_in)
    ctx.add_layer(ys1)
    ctx.add_layer(ys2_layer)
    ctx.add_layer(mixed_layer)
    ctx.add_layer(cameras)
    
    print(ctx.description())
    
    # Run
    results = ctx.observe()
    
    # Verification
    print(f"\nResults type: {type(results)}")
    print(f"Number of outputs: {len(results)}")
    
    assert len(results) == 5, f"Expected 5 outputs, got {len(results)}"
    
    # Check energy/amplitude flow
    # Input energy = 1 (normalized)
    # After YS1: 2 outputs, each 0.5 intensity
    # After YS2 layer: 4 outputs, each 0.25 intensity
    # Mixed layer:
    # - Path 0 (Pass): 0.25
    # - Path 1, 2 (MMI): Inputs are 0.25, 0.25. MMI mixes them.
    # - Path 3 (Pass): 0.25
    
    # Check output shapes
    for i, res in enumerate(results):
        print(f"Output {i} shape: {res.shape}")
        assert res.shape == (16, 16)
        
    print("\nPipeline verification successful!")

if __name__ == "__main__":
    test_pipeline()
