# Scaled Angular Spectrum Method (S-ASM)

The Scaled Angular Spectrum Method (S-ASM) is an advanced variation of ASM that allows for **arbitrary output window sizes** and resolutions, overcoming the fixed-grid limitation of standard ASM.

## Physics

S-ASM relies on the same exact solution to the Helmholtz equation as ASM:

$$ A(k_x, k_y) \rightarrow A(k_x, k_y) e^{izk_z} $$

However, instead of using the FFT (which restricts the frequency sampling to $1/L$), S-ASM typically employs the **Matrix Fourier Transform (MFT)** or Chirp Z-Transform (CZT).

By using MFT, we can evaluate the Fourier Transform at arbitrary output frequencies and spatial coordinates. This allows us to "zoom in" on a region of interest or change the sampling rate while maintaining the rigorous physics of the angular spectrum approach.

## Implementation

HELIOS implements S-ASM using Matrix Fourier Transforms.

1.  **Forward MFT**: Transform input field $U(x,y)$ to angular spectrum $A(f_x, f_y)$ on a custom frequency grid tailored to the output window.
2.  **Propagation**: Apply transfer function (phase shift).
3.  **Backward MFT**: Transform back to spatial domain on the requested output grid $U'(x', y')$.

*   **Complexity**: $O(N^3)$ (matrix multiplication) vs $O(N^2 \log N)$ for FFT. It is slower than standard ASM for large arrays but much more flexible.
*   **Benefits**: Exact physics + Zoom capability.

## Usage

S-ASM is ideal for:
*   High-accuracy propagation to a region of interest (ROI) smaller than the full grid.
*   Situations requiring non-standard pixel scales in the near/intermediate field.

```python
# Propagate with zooming (change output size)
wf = wf.propagate(10*u.cm, output_size=1*u.mm, regime='scasm')
```
