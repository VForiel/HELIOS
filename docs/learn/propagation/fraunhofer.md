# Fraunhofer Propagation

Fraunhofer propagation is used to calculate the diffraction pattern in the **far-field** regime or at the **focal plane** of a lens.

## Physics

The Fraunhofer approximation assumes that the propagation distance $z$ is sufficiently large ($z \gg D^2/\lambda$) such that the curvature of the wavefront across the aperture can be approximated as planar (or quadratic if focusing).

Mathematically, the field $U_2(x, y)$ at the observation plane is related to the field $U_1(\xi, \eta)$ at the aperture plane by a Fourier Transform:

$$
U_2(x, y) = \frac{e^{ikz} e^{i \frac{k}{2z}(x^2+y^2)}}{i \lambda z} \iint_{-\infty}^{\infty} U_1(\xi, \eta) e^{-i \frac{2\pi}{\lambda z} (x\xi + y\eta)} \,d\xi\,d\eta
$$

In the focal plane of a lens of focal length $f$, the quadratic phase term $e^{i \frac{k}{2z}(x^2+y^2)}$ is cancelled by the lens, and the relationship becomes an exact Fourier Transform (scaled by $\lambda f$).

## Implementation

In HELIOS, `Fraunhofer` is implemented using a Fast Fourier Transform (FFT).

*   **Algorithm**: FFT-based.
*   **Complexity**: $O(N \log N)$.
*   **Grid**: The output grid resolution is determined by the diffraction reciprocity relation:
    $$ \Delta x_{out} = \frac{\lambda z}{N \Delta x_{in}} $$
    where $N$ is the number of pixels.

To obtain a specific output window size or resolution different from the natural FFT grid, HELIOS employs zero-padding or cropping of the output array.

## Usage

This method is automatically selected by `Wavefront.propagate()` when:
*   The propagation is to the geometric focus of a lens ($z = f$).
*   The user explicitly sets `regime='fraunhofer'`.

```python
# Force Fraunhofer propagation
wf_out = wf.propagate(100*u.m, regime='fraunhofer')
```
