"""
Demonstration of realistic camera observation with detector noise.

This example shows:
1. Simple star observation through a pupil (diffraction pattern)
2. Camera detector physics: dark current, shot noise, read noise
3. Triple plot: raw image, dark frame, reduced image
4. Temporal plot: signal accumulation over time with noise contributions
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import helios


def demo_camera_observation():
    """
    Simulate a simple star observation with realistic detector noise.
    """
    print("=" * 60)
    print("HELIOS Camera Observation Demo")
    print("=" * 60)
    
    # =========================================================================
    # 1. Setup: Scene, Telescope, Camera
    # =========================================================================
    
    # Create a distant scene to reduce flux (1000 pc instead of default 10 pc)
    scene = helios.Scene(distance=2_000*u.pc)
    scene.add(helios.Star(
        position=(0*u.arcsec, 0*u.arcsec),
        temperature=5778*u.K,  # Sun-like star
        magnitude=5.0,
        name="Distant Star"
    ))
    
    # Create very small telescope for large, visible Airy pattern
    # Smaller aperture → Larger PSF with visible Airy rings
    pupil = helios.Pupil(diameter=0.01*u.m)
    pupil.add_disk(radius=0.005)  # 1cm diameter aperture (tiny telescope)
    telescope = helios.TelescopeArray(name="Tiny Telescope (1cm)")
    telescope.add_collector(pupil=pupil, position=(0, 0), size=0.01*u.m)
    
    # Instrument temperature for thermal emission
    instrument_temp = 280*u.K  # Warm instrument thermal emission
    
    # Create camera with realistic detector parameters
    camera = helios.Camera(
        pixels=(256, 256),
        pixel_size=13*u.um,
        integration_time=0.5*u.s,  # 500ms
        dark_current=500*u.electron/u.s,  # Very high dark current >> signal
        read_noise=1*u.electron,  # Low read noise (modern sCMOS/EMCCD)
        quantum_efficiency=0.9,
        thermal_background_temp=instrument_temp,  # Thermal emission from warm instrument
        name="Science Camera"
    )
    
    # =========================================================================
    # 2. Create Context and Observe
    # =========================================================================
    
    ctx = helios.Context()
    ctx.add_layer(scene)
    ctx.add_layer(telescope)
    ctx.add_layer(camera)

    print(ctx.description(full=True))
    
    print("\n" + "=" * 60)
    print("Running simulation...")
    print("=" * 60)
    
    # Process wavefront through scene and telescope to get PSF at focal plane
    wf = helios.Wavefront(wavelength=550*u.nm, size=512)
    wf = scene.process(wf, ctx)
    wf = telescope.process(wf, ctx)
    
    # CRITICAL: Propagate to focal plane to create diffraction pattern (PSF)
    # The wavefront is currently in pupil plane, we need FFT to get image plane
    wf.field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wf.field)))
    
    # Get raw image and dark frame from camera
    # Note: thermal background is now automatically included in camera.get_raw_image()
    raw_image = camera.get_raw_image(wf, ctx)
    dark_frame = camera.get_dark()
    
    # Compute reduced image by subtracting dark from raw
    reduced_image = raw_image - dark_frame
    
    # DIAGNOSTIC: Check if we actually have signal in the wavefront
    intensity = np.abs(wf.field) ** 2
    print(f"\nDIAGNOSTIC:")
    print(f"Wavefront intensity: min={intensity.min():.2e}, max={intensity.max():.2e}, mean={intensity.mean():.2e}")
    print(f"Wavefront field amplitude: min={np.abs(wf.field).min():.2e}, max={np.abs(wf.field).max():.2e}")
    
    print(f"\nRaw image shape: {raw_image.shape}")
    print(f"Raw image: min={raw_image.min():.1f}, max={raw_image.max():.1f}, mean={raw_image.mean():.1f} e-")
    print(f"Dark frame: min={dark_frame.min():.1f}, max={dark_frame.max():.1f}, mean={dark_frame.mean():.1f} e-")
    print(f"Reduced image: min={reduced_image.min():.1f}, max={reduced_image.max():.1f}, mean={reduced_image.mean():.1f} e-")
    
    # Check if signal is actually present
    signal_only = intensity * camera.quantum_efficiency * camera.integration_time_value
    if signal_only.shape != camera.pixels:
        from scipy.ndimage import zoom
        zoom_factors = (camera.pixels[0]/signal_only.shape[0], camera.pixels[1]/signal_only.shape[1])
        signal_only = zoom(signal_only, zoom_factors, order=1)
    print(f"Expected signal (no noise): min={signal_only.min():.1f}, max={signal_only.max():.1f}, sum={signal_only.sum():.1f} e-")
    
    # =========================================================================
    # 3. Triple Plot: Raw, Dark, Reduced (2 rows: linear + log scale)
    # =========================================================================
    
    # NOTE: 2x3 grid - top row linear, bottom row log scale
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    vmin_log = 0.1  # Minimum value for log scale
    vmax_raw = raw_image.max()
    vmax_dark = max(dark_frame.max(), 1)
    
    # ========== TOP ROW: LINEAR SCALE ==========
    
    # Raw image (linear)
    im0_lin = axes[0, 0].imshow(raw_image, origin='lower', cmap='hot', vmin=0, vmax=vmax_raw)
    axes[0, 0].set_title(f'Raw Image (linear)\\nMax={raw_image.max():.0f} e-', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('X [pixels]')
    axes[0, 0].set_ylabel('Y [pixels]')
    plt.colorbar(im0_lin, ax=axes[0, 0], label='Electrons', fraction=0.046, pad=0.04)
    
    # Dark frame (linear)
    im1_lin = axes[0, 1].imshow(dark_frame, origin='lower', cmap='gray', vmin=0, vmax=vmax_dark)
    axes[0, 1].set_title(f'Dark Frame (linear)\\nMax={dark_frame.max():.0f} e-', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('X [pixels]')
    axes[0, 1].set_ylabel('Y [pixels]')
    plt.colorbar(im1_lin, ax=axes[0, 1], label='Electrons', fraction=0.046, pad=0.04)
    
    # Reduced image (linear)
    im2_lin = axes[0, 2].imshow(reduced_image, origin='lower', cmap='hot', vmin=reduced_image.min(), vmax=vmax_raw)
    axes[0, 2].set_title(f'Reduced Image (linear)\\nMax={reduced_image.max():.0f} e-', fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel('X [pixels]')
    axes[0, 2].set_ylabel('Y [pixels]')
    plt.colorbar(im2_lin, ax=axes[0, 2], label='Electrons', fraction=0.046, pad=0.04)
    
    # ========== BOTTOM ROW: LOG SCALE ==========
    
    # Raw image (log)
    im0_log = axes[1, 0].imshow(np.maximum(raw_image, vmin_log), origin='lower', cmap='hot', 
                                 norm=plt.matplotlib.colors.LogNorm(vmin=vmin_log, vmax=vmax_raw))
    axes[1, 0].set_title(f'Raw Image (log scale)\\nMax={raw_image.max():.0f} e-', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('X [pixels]')
    axes[1, 0].set_ylabel('Y [pixels]')
    plt.colorbar(im0_log, ax=axes[1, 0], label='Electrons', fraction=0.046, pad=0.04)
    
    # Dark frame (log)
    im1_log = axes[1, 1].imshow(np.maximum(np.abs(dark_frame), vmin_log), origin='lower', cmap='gray',
                                 norm=plt.matplotlib.colors.LogNorm(vmin=vmin_log, vmax=vmax_dark))
    axes[1, 1].set_title(f'Dark Frame (log |value|)\\nMax={dark_frame.max():.0f} e-', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('X [pixels]')
    axes[1, 1].set_ylabel('Y [pixels]')
    plt.colorbar(im1_log, ax=axes[1, 1], label='|Electrons|', fraction=0.046, pad=0.04)
    
    # Reduced image (log)
    im2_log = axes[1, 2].imshow(np.maximum(reduced_image, vmin_log), origin='lower', cmap='hot',
                                 norm=plt.matplotlib.colors.LogNorm(vmin=vmin_log, vmax=vmax_raw))
    axes[1, 2].set_title(f'Reduced Image (log scale)\\nMax={reduced_image.max():.0f} e-', fontsize=11, fontweight='bold')
    axes[1, 2].set_xlabel('X [pixels]')
    axes[1, 2].set_ylabel('Y [pixels]')
    plt.colorbar(im2_log, ax=axes[1, 2], label='Electrons', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('camera_observation_triple_plot.png', dpi=150, bbox_inches='tight')
    print("\n✓ Triple plot (2 rows: linear + log) saved as 'camera_observation_triple_plot.png'")
    
    # =========================================================================
    # 4. Temporal Analysis: Signal Accumulation Over Time
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Computing temporal signal accumulation...")
    print("=" * 60)
    
    # Simulate integration over time
    max_time = float(camera.integration_time.to(u.s).value)
    time_points = np.linspace(0, max_time, 100)
    
    # Get the signal intensity (no noise) from the wavefront
    intensity = np.abs(wf.field) ** 2
    
    # Resize to camera pixels if needed
    if intensity.shape != camera.pixels:
        from scipy.ndimage import zoom
        zoom_factors = (camera.pixels[0] / intensity.shape[0], 
                       camera.pixels[1] / intensity.shape[1])
        intensity = zoom(intensity, zoom_factors, order=1)
    
    # Convert to electrons per second
    signal_rate = intensity * camera.quantum_efficiency  # e-/s per pixel
    dark_rate = camera.dark_current  # e-/s per pixel (already a float)
    thermal_rate = camera.thermal_background  # e-/s per pixel (from camera)
    
    # Total signal and background integrated over time
    total_signal = np.sum(signal_rate)  # Total e-/s from all pixels
    total_dark_rate = dark_rate * np.prod(camera.pixels)  # Total dark e-/s
    total_thermal_rate = thermal_rate * np.prod(camera.pixels)  # Total thermal e-/s
    
    # Accumulated electrons over time
    signal_electrons = total_signal * time_points
    dark_electrons = total_dark_rate * time_points
    thermal_electrons = total_thermal_rate * time_points
    total_electrons = signal_electrons + dark_electrons + thermal_electrons
    
    # Shot noise (Poisson) grows as sqrt(signal)
    # Read noise is added at the end (independent of integration time)
    shot_noise_sigma = np.sqrt(total_electrons)  # Standard deviation
    read_noise_total = float(camera.read_noise) * np.sqrt(np.prod(camera.pixels))
    
    print(f"\nSignal rate: {total_signal:.2e} e-/s")
    print(f"Dark rate: {total_dark_rate:.2e} e-/s")
    print(f"Thermal rate: {total_thermal_rate:.2e} e-/s")
    total_background_rate = total_dark_rate + total_thermal_rate
    print(f"Signal/Total background ratio: {total_signal/total_background_rate:.1f}")
    print(f"\nAt t={max_time}s:")
    print(f"  Signal: {signal_electrons[-1]:.2e} e-")
    print(f"  Dark: {dark_electrons[-1]:.2e} e-")
    print(f"  Thermal: {thermal_electrons[-1]:.2e} e-")
    print(f"  Shot noise (σ): {shot_noise_sigma[-1]:.2e} e-")
    print(f"  Read noise (σ): {read_noise_total:.2e} e-")
    print(f"  SNR: {signal_electrons[-1] / np.sqrt(total_electrons[-1] + read_noise_total**2):.1f}")
    
    # Create temporal plot
    # NOTE: Using smaller figure size to fit on screen
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    
    # Top panel: Accumulated electrons
    axes[0].plot(time_points, signal_electrons, 'b-', linewidth=2, label='Signal (Star)')
    axes[0].plot(time_points, dark_electrons, 'r--', linewidth=2, label='Dark Current')
    axes[0].plot(time_points, thermal_electrons, 'orange', linestyle='--', linewidth=2, label='Thermal (Instrument)')
    axes[0].plot(time_points, total_electrons, 'k-', linewidth=2.5, label='Total')
    axes[0].axhline(read_noise_total, color='gray', linestyle=':', linewidth=2, label=f'Read Noise (σ={read_noise_total:.1e} e-)')
    axes[0].set_ylabel('Accumulated Electrons', fontsize=12, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Signal Accumulation Over Time (All Noise Sources)', fontsize=14, fontweight='bold')
    
    # Bottom panel: Noise evolution
    axes[1].plot(time_points, shot_noise_sigma, 'g-', linewidth=2, label='Shot Noise (√Signal)')
    axes[1].axhline(read_noise_total, color='orange', linestyle=':', linewidth=2, label=f'Read Noise (constant)')
    total_noise = np.sqrt(shot_noise_sigma**2 + read_noise_total**2)
    axes[1].plot(time_points, total_noise, 'k-', linewidth=2, label='Total Noise (quadrature sum)')
    axes[1].set_xlabel('Integration Time [s]', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Noise (σ) [electrons]', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('camera_observation_temporal.png', dpi=150, bbox_inches='tight')
    print("\n✓ Temporal plot saved as 'camera_observation_temporal.png'")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo_camera_observation()
