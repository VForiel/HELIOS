import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import numba as nb
import warnings

def _crop_or_pad(array: np.ndarray, size: int) -> np.ndarray:
    """
    Resize a 2D array to a target size by cropping or zero-padding.
    Centers the result.
    """
    curr_size = array.shape[0]
    if curr_size == size: return array
    
    if curr_size > size:
        start = (curr_size - size) // 2
        return array[start:start+size, start:start+size]
    else:
        pad = (size - curr_size) // 2
        return np.pad(array, pad, mode='constant')

def fraunhofer(ψ0, L0, λ, z, Lf=None, Nf=None, verbose=False):
    """
    Fraunhofer propagation (Far-field / Focal Plane).
    Uses a single FFT with energy conservation factors.
    """
    N0 = ψ0.shape[0]
    if Nf is None: Nf = N0
    dx_in = L0 / N0
    
    # In Fraunhofer, Lf is fixed by physics: Lf = lambda * z * N / L0
    # If user requests specific Lf/Nf, we handle it via padding/cropping output.
    
    k = 2 * np.pi / λ
    
    # 1. FFT
    # Note: Using 'norm="ortho"' helps with energy but standard definition is preferred for physics transparency
    # Shift -> FFT -> Shift
    spectrum = fftshift(fft2(ifftshift(ψ0)))
    
    # 2. Physics Scaling
    # Field = (1 / i*lam*z) * exp(ikz) * exp(ik(x^2+y^2)/2z) * FFT( U * ...)
    # In strict Fraunhofer, the quadratic phase term exp(ik r^2 / 2z) is often neglected 
    # for intensity, but we keep the pre-factors for energy conservation.
    
    # FFT approximation factor: Integral ~ Sum * dx * dy
    scaling_factor = (dx_in ** 2) / (1j * λ * z) * np.exp(1j * k * z)
    
    ψ_out = spectrum * scaling_factor
    
    # 3. Resize to target resolution
    return _crop_or_pad(ψ_out, Nf)

def fresnel(ψ0, L0, λ, z, Lf=None, Nf=None, verbose=False):
    """
    Fresnel propagation using single FFT (Impulse Response formulation).
    Best for focusing or intermediate distances.
    """
    if verbose: print("--- Fresnel FFT Propagation ---")

    N0 = ψ0.shape[0]
    dx_in = L0 / N0
    if Nf is None: Nf = N0
    
    # Calculate required padding to achieve target resolution Lf/Nf
    if Lf is not None:
        dx_target = Lf / Nf
        # Critical sampling condition for chirp
        val = (λ * abs(z)) / (dx_in * dx_target)
        N_required = int(np.round(val))
        if N_required < N0: N_required = N0
    else:
        N_required = N0

    pad_size = (N_required - N0) // 2
    N_padded = N0 + 2 * pad_size
    field_padded = np.pad(ψ0, pad_size, mode='constant')
    
    # Coordinates for chirp
    x = (np.arange(N_padded) - N_padded // 2) * dx_in
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2
    k = 2 * np.pi / λ
    
    # 1. Input Quadratic Phase
    Q1 = np.exp(1j * k * R2 / (2 * z))
    
    # 2. FFT
    # Integral approx factor included here
    pre_factor = (dx_in ** 2) / (1j * λ * z) * np.exp(1j * k * z)
    field_fft = fftshift(fft2(ifftshift(field_padded * Q1)))
    
    # 3. Output Quadratic Phase
    dx_out_actual = (λ * abs(z)) / (N_padded * dx_in)
    x_out = (np.arange(N_padded) - N_padded // 2) * dx_out_actual
    X_out, Y_out = np.meshgrid(x_out, x_out)
    R2_out = X_out**2 + Y_out**2
    
    Q2 = np.exp(1j * k * R2_out / (2 * z))
    
    ψ_full = field_fft * Q2 * pre_factor
    
    # 4. Resize to requested Nf
    return _crop_or_pad(ψ_full, Nf)

def asm(ψ0, L0, λ, z, Lf=None, Nf=None, verbose=False):
    """
    Angular Spectrum Method (ASM).
    Exact scalar solution. Uses FFT. Preserves pixel scale (dx_out = dx_in).
    """
    if verbose: print("--- Angular Spectrum Method (FFT-based) ---")
    
    N = ψ0.shape[0]
    if Nf is None: Nf = N
    dx = L0 / N
    k = 2 * np.pi / λ
    
    # 1. Frequency Grid
    freq = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(freq, freq)
    kx = 2 * np.pi * KX
    ky = 2 * np.pi * KY
    
    # 2. Forward FFT
    A0 = np.fft.fft2(ψ0)
    
    # 3. Transfer Function
    # kz = sqrt(k^2 - kx^2 - ky^2)
    kz_sq = k**2 - kx**2 - ky**2
    
    # Evanescent wave mask (remove non-physical high frequencies)
    mask = kz_sq >= 0
    kz = np.zeros_like(kz_sq)
    kz[mask] = np.sqrt(kz_sq[mask])
    
    # Propagator H = exp(i * kz * z)
    H = np.exp(1j * kz * z)
    H[~mask] = 0 # Kill evanescent waves
    
    # 4. Propagation & Inverse FFT
    Az = A0 * H
    ψz = np.fft.ifft2(Az)
    
    # ASM maintains grid size. If Nf != N, we crop/pad
    return _crop_or_pad(ψz, Nf)

def mft_matrix(r_in, r_out, mode='forward'):
    """Helper for Matrix Fourier Transform."""
    sign = -1j if mode == 'forward' else 1j
    return np.exp(sign * 2 * np.pi * np.outer(r_out, r_in))

def scasm(ψ0, L0, λ, z, Lf=None, Nf=None, verbose=False):
    """
    Scaled Angular Spectrum Method (S-ASM) via MFT.
    Allows changing window size (zooming) with exact physics.
    """
    if verbose: print("--- Scaled ASM (MFT) ---")
    
    Ny, Nx = ψ0.shape
    if Nf is None: Nf = Nx
    if Lf is None: Lf = L0
    
    # Spatial Grids
    dx0 = L0 / Nx
    x0 = np.linspace(-L0/2, L0/2 - dx0, Nx)
    y0 = np.linspace(-L0/2, L0/2 - dx0, Ny)
    
    dxf = Lf / Nf
    xf = np.linspace(-Lf/2, Lf/2 - dxf, Nf)
    yf = np.linspace(-Lf/2, Lf/2 - dxf, Nf)
    
    # Frequency Grid (Bandwidth limited by input resolution)
    f_max_x = 1.0 / (2 * dx0)
    f_max_y = 1.0 / (2 * (L0/Ny))
    fx = np.linspace(-f_max_x, f_max_x, Nx)
    fy = np.linspace(-f_max_y, f_max_y, Ny)
    FX, FY = np.meshgrid(fx, fy)
    
    # 1. Transform to Frequency Domain (MFT)
    Mx_fwd = mft_matrix(x0, fx, mode='forward')
    My_fwd = mft_matrix(y0, fy, mode='forward')
    # Scale factor dx*dy for integral approx
    spectrum = np.matmul(My_fwd, np.matmul(ψ0, Mx_fwd.T)) * (dx0 * (L0/Ny))

    # 2. Apply Transfer Function
    k = 2 * np.pi / λ
    arg_sqrt = (1/λ)**2 - (FX**2 + FY**2)
    mask = arg_sqrt >= 0
    KZ = np.zeros_like(arg_sqrt)
    KZ[mask] = 2 * np.pi * np.sqrt(arg_sqrt[mask])
    
    Transfer = np.exp(1j * KZ * z)
    Transfer[~mask] = 0
    spectrum_prop = spectrum * Transfer
    
    # 3. Transform back to Space (MFT)
    Mx_bwd = mft_matrix(fx, xf, mode='backward')
    My_bwd = mft_matrix(fy, yf, mode='backward')
    
    dfx = fx[1] - fx[0]
    dfy = fy[1] - fy[0]
    ψz = np.matmul(My_bwd, np.matmul(spectrum_prop, Mx_bwd.T)) * (dfx * dfy)
    
    return ψz

def rs_direct(ψ0, L0, λ, z, Lf=None, Nf=None, verbose=False):
    """
    Rayleigh-Sommerfeld Direct Integration.
    Exact vector/scalar solution. O(N^4). Reference only.
    """
    if verbose: print("--- Rayleigh-Sommerfeld Direct ---")
    Ny, Nx = ψ0.shape
    if Nf is None: Nf = Nx
    if Lf is None: Lf = L0
    
    dx0 = L0 / Nx
    dxf = Lf / Nf
    
    # Source Coordinates
    xs = np.linspace(-L0/2, L0/2 - dx0, Nx)
    ys = np.linspace(-L0/2, L0/2 - dx0, Ny)
    XS, YS = np.meshgrid(xs, ys)
    flat_source_x = XS.flatten()
    flat_source_y = YS.flatten()
    flat_U0 = ψ0.flatten()
    
    # Target Coordinates
    xf = np.linspace(-Lf/2, Lf/2 - dxf, Nf)
    yf = np.linspace(-Lf/2, Lf/2 - dxf, Nf)
    XT, YT = np.meshgrid(xf, yf)
    flat_target_x = XT.flatten()
    flat_target_y = YT.flatten()
    
    flat_Uz = np.zeros(flat_target_x.size, dtype=complex)
    k = 2 * np.pi / λ
    
    # Batch processing to save RAM
    block_size = 500 
    
    for i in range(0, flat_target_x.size, block_size):
        end = min(i + block_size, flat_target_x.size)
        tx = flat_target_x[i:end]
        ty = flat_target_y[i:end]
        
        # Broadcasting: (Batch, Sources)
        dx_mat = tx[:, np.newaxis] - flat_source_x[np.newaxis, :]
        dy_mat = ty[:, np.newaxis] - flat_source_y[np.newaxis, :]
        
        R2 = dx_mat**2 + dy_mat**2 + z**2
        R = np.sqrt(R2)
        
        # RS1 Kernel: (1/2pi) * (z/R) * (exp(ikR)/R) * (ik - 1/R)
        # Often approx as (1/i*lam) * (z/R) * exp(ikR)/R in optics
        # We use standard optic approximation k >> 1/R
        pre_factor = (1 / (1j * λ)) * (z / R**2) * np.exp(1j * k * R)
        
        flat_Uz[i:end] = np.sum(pre_factor * flat_U0[np.newaxis, :], axis=1)
        
    return flat_Uz.reshape(Nf, Nf) * (dx0**2)

@nb.jit(nopython=True, parallel=True)
def _fresnel_custom_kernel(ψ0, Δx0, λ, z, Δxf, Nf):
    """Numba kernel for direct Fresnel sum."""
    π = np.pi
    k = 2 * π / λ
    i = 1j
    N0 = len(ψ0)
    
    # Precompute Source Grid
    X = np.empty((N0, N0), dtype=np.float64)
    Y = np.empty((N0, N0), dtype=np.float64)
    for m in range(N0):
        val = Δx0 * (m - (N0-1)/2.0)
        X[:, m] = val
        Y[m, :] = -val # Flip Y conventionally
        
    ψz = np.zeros((Nf, Nf), dtype=np.complex128)
    pre = (np.exp(i*k*z) / (i * λ * z)) * (Δx0**2)
    coeff = i*k/(2*z)

    for xi in nb.prange(Nf):
        x = Δxf * (xi - (Nf-1)/2.0)
        for yi in range(Nf):
            y = Δxf * (yi - (Nf-1)/2.0)
            
            # Integral sum
            sum_val = 0.0 + 0.0j
            for r in range(N0):
                for c in range(N0):
                    phase = coeff * ((x - X[r,c])**2 + (y - Y[r,c])**2)
                    sum_val += ψ0[r,c] * np.exp(phase)
            
            ψz[xi, yi] = pre * sum_val
    return ψz

def fresnel_custom(ψ0, L0, λ, z, Lf=None, Nf=None, verbose=False):
    """Wrapper for Numba Fresnel."""
    if verbose: print("--- Fresnel Custom (Numba) ---")
    N0 = ψ0.shape[0]
    if Nf is None: Nf = N0
    if Lf is None: Lf = L0
    return _fresnel_custom_kernel(ψ0, L0/N0, λ, z, Lf/Nf, Nf)

#==============================================================================
# Demo
#==============================================================================

if __name__ == "__main__":
    N0 = 100
    Nf = N0
    L0 = 1
    Lf = 1
    f = 100
    z = 100

    λ=550e-9

    pupil = helios.Pupil.jwst()
    ψ0 = pupil.get_array(N0, oversample=4).astype(np.complex128)

    # Lens
    k = 2 * π / λ
    x = (np.arange(N0) - N0 / 2) * L0 / N0
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2
    lens_phase = np.exp(-1j * k * R2 / (2 * f))
    ψ0 *= lens_phase

    Δx0 = L0 / N0
    Δxf = Lf / Nf

    for method in ['frauhofer', 'fresnel', 'fresnel_custom', 'asm', 'scasm', 'rs_direct']:
        

        ψz = method(ψ0, L0, λ, z, Lf, Nf, verbose=True)

        print("Energy gain:", np.sum(np.abs(ψz)**2) / np.sum(np.abs(ψ0)**2))

        fig, axs = plt.subplots(1,2)
        im = axs[0].imshow(np.abs(ψ0)**2)
        fig.colorbar(im, ax=axs[0])
        axs[1].imshow(np.abs(ψz)**2)
        fig.colorbar(im, ax=axs[1])
        axs[0].set_title(f"Initial (size = {L0:.2e} m)")
        axs[1].set_title(f"Final (size = {Lf:.2e} m)")
        plt.show()
