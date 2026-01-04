# Rayleigh-Sommerfeld Direct Integration

Rayleigh-Sommerfeld (RS) Direct Integration is the **numerical gold standard** for diffraction calculation. It involves directly summing the contributions of spherical waves emitted from every point in the source aperture.

## Physics

The RS diffraction formula (specifically the first Rayleigh-Sommerfeld integral) is:

$$
U_2(x, y) = \frac{1}{i\lambda} \iint_{\Sigma} U_1(\xi, \eta) \frac{z}{R} \frac{e^{ikR}}{R} \,d\xi\,d\eta
$$

where $R = \sqrt{(x-\xi)^2 + (y-\eta)^2 + z^2}$ is the distance from a source point $(\xi, \eta)$ to a target point $(x, y)$.

Unlike Fourier-based methods, this formulation makes **no paraxial approximation** and is valid for any geometry.

## Implementation

HELIOS implements this via a direct summation loop (or matrix broadcasting):

1.  For every pixel in the output grid $(x_i, y_i)$.
2.  Sum the contributions from all pixels in the input grid $(\xi_j, \eta_j)$.

*   **Complexity**: $O(N_x N_y \times M_x M_y) \approx O(N^4)$. This is **extremely computationally expensive**.
*   **Accuracy**: Highest possible accuracy (limited only by discretization).

## Usage

Due to its high cost, RS-Direct is primarily used for:
*   Reference validation of faster methods (Fraunhofer, Fresnel, ASM).
*   Very small arrays or sparse calculation points.
*   Geometries where other approximations fail completely.

```python
# Use only for small arrays!
wf_out = wf.propagate(10*u.cm, regime='rs_direct')
```
