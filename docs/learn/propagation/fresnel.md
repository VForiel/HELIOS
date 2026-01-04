# Fresnel Propagation

Fresnel propagation is used for **near-field** to **intermediate-field** diffraction, where the paraxial approximation holds but the distance is not large enough for the Fraunhofer approximation.

## Physics

The Fresnel diffraction integral is given by:

$$
U_2(x, y) = \frac{e^{ikz}}{i \lambda z} \iint_{-\infty}^{\infty} U_1(\xi, \eta) e^{i \frac{k}{2z} [(x-\xi)^2 + (y-\eta)^2]} \,d\xi\,d\eta
$$

This can be rewritten as a convolution of the input field with a quadratic phase kernel (the free-space impulse response):

$$
U_2(x, y) = U_1(x, y) * h(x, y) \quad \text{where} \quad h(x, y) = \frac{e^{ikz}}{i \lambda z} e^{i \frac{k}{2z}(x^2+y^2)}
$$

## Implementation

HELIOS implements Fresnel propagation using the **Single Fourier Transform** (or Impulse Response) method. This approach involves:

1.  Multiplying the input field by a quadratic phase factor.
2.  Taking the FFT.
3.  Multiplying by another quadratic phase factor.

This formulation allows for flexible output grid scaling, making it suitable for focusing beams where the output window needs to shrink as the beam converges.

*   **Algorithm**: FFT-based (Impulse Response).
*   **Complexity**: $O(N \log N)$.
*   **Sampling Condition**: To avoid aliasing of the quadratic phase chirp, the sampling must satisfy:
    $$ N \ge \frac{\lambda z}{\Delta x_{in} \Delta x_{out}} $$
    If this condition is not met (e.g., at very short distances), HELIOS automatically zero-pads the input array to increase $N$ sufficiently.

## Usage

This method is suitable for:
*   Propagation over moderate distances.
*   Converging beams (e.g. after a lens) before the focal plane.

```python
# Automatic selection handles this at appropriate distances
wf = wf.propagate(10*u.cm)

# Or force it explicitly
wf = wf.propagate(10*u.cm, regime='fresnel')
```
