# Angular Spectrum Method (ASM)

The Angular Spectrum Method (ASM) provides an **exact scalar solution** to the Helmholtz equation, valid for all propagation distances, including the very near field.

## Physics

ASM works by decomposing the wavefront into a superposition of plane waves, propagating each plane wave by its corresponding phase shift, and then recombining them.

1.  **Decomposition**: Compute the angular spectrum $A(k_x, k_y)$ of the input field $U(x, y)$ via FFT.
2.  **Propagation**: Multiply by the Transfer Function $H(k_x, k_y)$:
    $$ H(k_x, k_y) = \exp\left(i z \sqrt{k^2 - k_x^2 - k_y^2}\right) $$
3.  **Reconstruction**: Compute the output field via Inverse FFT.

**Evanescent Waves**: For high spatial frequencies where $k_x^2 + k_y^2 > k^2$, the square root becomes imaginary. These components decay exponentially (evanescent waves) and do not propagate to the far field. HELIOS typically filters these out ($H=0$).

## Implementation

*   **Algorithm**: FFT-based (Transfer Function).
*   **Complexity**: $O(N \log N)$.
*   **Resolution**: ASM naturally preserves the sampling resolution ($ \Delta x_{out} = \Delta x_{in} $). The output window size matches the input window size.

## Usage

ASM is the preferred method for:
*   Short propagation distances.
*   Situations where maintaining the same spatial grid is desired.

Since ASM requires the output pixel scale to match the input, it is less suitable for focusing beams where the beam size changes dramatically, unless immense zero-padding is used (which is computationally expensive).

```python
# Force ASM propagation
wf_out = wf.propagate(1*u.mm, regime='asm')
```
