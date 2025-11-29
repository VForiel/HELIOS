import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Prefer project `src` directory so tests exercise the local (edited) code
here = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(here, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from helios.components.coronagraph import Coronagraph


def test_coronagraph_mask_array_and_plot():
    # create a coronagraph and request mask
    coro = Coronagraph(phase_mask='4quadrants')
    mask = coro.mask_array(npix=128)
    # mask must be complex and square
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (128, 128)
    assert np.iscomplexobj(mask)

    # plotting should not raise and should return an Axes
    ax = coro.plot_mask(npix=128)
    assert hasattr(ax, 'imshow')
    plt.close(ax.figure)

def test_vortex_phase_plot():
    coro = Coronagraph(phase_mask='vortex')
    mask = coro.mask_array(npix=64, kind='vortex', charge=2)
    assert mask.shape == (64, 64)
    ax = coro.plot_mask(npix=64, kind='vortex', charge=2)
    assert hasattr(ax, 'imshow')
    plt.close(ax.figure)
