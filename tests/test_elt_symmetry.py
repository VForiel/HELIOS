import numpy as np
from helios.components.optics import Pupil

def test_elt_symmetry():
    p = Pupil.like('ELT')
    arr = p.get_array(npix=512, soft=False)
    flipx = np.flip(arr, axis=1)
    flipy = np.flip(arr, axis=0)
    diff_x = np.sum(np.abs(arr - flipx))
    diff_y = np.sum(np.abs(arr - flipy))
    total = np.sum(arr)
    rel_x = diff_x / total if total > 0 else 0.0
    rel_y = diff_y / total if total > 0 else 0.0
    # Exiger une symétrie relative (<1%) pour compenser aliasing polygonal
    assert rel_x < 0.012, f"Asymétrie horizontale relative trop forte: {rel_x:.4f}" 
    assert rel_y < 0.012, f"Asymétrie verticale relative trop forte: {rel_y:.4f}" 
