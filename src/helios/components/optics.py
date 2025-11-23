import numpy as np
from astropy import units as u
from typing import Tuple, List, Union, Optional
from ..core.context import Layer, Context
from ..core.simulation import Wavefront
from matplotlib.path import Path
import matplotlib.pyplot as _plt


class Pupil:
    """Pupil builder and rasterizer.

    - You can add primitive elements (disk, hexagon, spiders, central obstruction)
    - Retrieve the final pupil as a NumPy array via `get_array(npix, soft=False)`
    - Soft edges are obtained with simple supersampling (anti-aliasing)
    - Presets: `Pupil.jwst()`, `Pupil.vlt()`, `Pupil.elt()` provide realistic basic configs
    """

    def __init__(self, diameter: u.Quantity = 1.0 * u.m):
        self.diameter = diameter.to(u.m).value
        self.elements: List[dict] = []

    # --- element adders -------------------------------------------------
    def add_disk(self, radius: float, center: Tuple[float, float] = (0.0, 0.0), value: float = 1.0):
        """Add a filled disk (radius in same units as `diameter`)."""
        self.elements.append({"type": "disk", "radius": float(radius), "center": tuple(center), "value": float(value)})

    def add_hexagon(self, radius: float, center: Tuple[float, float] = (0.0, 0.0), value: float = 1.0, rotation: float = 0.0):
        """Add a regular hexagon (radius = circumradius)."""
        self.elements.append({"type": "hex", "radius": float(radius), "center": tuple(center), "value": float(value), "rotation": float(rotation)})

    def add_central_obscuration(self, diameter: float):
        """Add central obscuration (diameter)."""
        self.elements.append({"type": "secondary", "diameter": float(diameter)})

    def add_spiders(self, arms: int = 4, width: float = 0.01, angle: float = 0.0, angles: Optional[List[float]] = None):
        """Add radially extended rectangular spider vanes.

        - `width` is in same linear units as `diameter`.
        - `angle` is rotation offset in degrees.
        """
        entry = {"type": "spiders", "arms": int(arms), "width": float(width), "angle": float(angle)}
        if angles is not None:
            # store explicit angles (degrees)
            entry["angles"] = [float(a) for a in angles]
        self.elements.append(entry)

    def add_segmented_primary(self, seg_flat: float, rings: int = 2, rotation: float = 0.0, gap: float = 0.0):
        """Create a hexagonal segmented primary with given flat-to-flat segment size.

        - `seg_flat`: flat-to-flat size of one segment (same units as `diameter`).
        - `rings`: number of hex rings around center (rings=0 -> 1 segment)
        """
        self.elements.append({"type": "segments", "seg_flat": float(seg_flat), "rings": int(rings), "rotation": float(rotation), "gap": float(gap)})

    # --- masks/rasterization helpers ------------------------------------
    def _make_grid(self, npix: int, oversample: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
        # grid in meters centered at 0
        size_m = self.diameter
        N = npix * oversample
        half = size_m / 2.0
        xs = np.linspace(-half, half, N, endpoint=False) + (size_m / N) / 2.0
        ys = xs.copy()
        xg, yg = np.meshgrid(xs, ys)
        return xg, yg, size_m / N

    def _hex_verts(self, cx: float, cy: float, R: float, rotation: float = 0.0):
        thetas = np.linspace(0, 2 * np.pi, 7) + np.deg2rad(rotation)
        verts = np.column_stack((cx + R * np.cos(thetas), cy + R * np.sin(thetas)))
        return verts

    def _rasterize_path(self, path: Path, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
        pts = np.column_stack((xg.ravel(), yg.ravel()))
        mask = path.contains_points(pts)
        return mask.reshape(xg.shape)

    def get_array(self, npix: int = 256, soft: bool = False, oversample: int = 4) -> np.ndarray:
        """Return the pupil as a 2D NumPy array (values in [0,1]).

        - `npix`: desired output pixels (square array)
        - `soft`: if True use `oversample` subpixels to anti-alias edges
        """
        ov = oversample if soft and oversample >= 2 else 1
        xg, yg, dx = self._make_grid(npix, oversample=ov)
        im = np.zeros_like(xg, dtype=float)

        for el in self.elements:
            t = el.get("type")
            if t == "disk":
                cx, cy = el["center"]
                r = el["radius"]
                mask = ((xg - cx) ** 2 + (yg - cy) ** 2) <= (r ** 2)
                val = float(el.get("value", 1.0))
                if val < 1.0:
                    # occluding disk: overwrite
                    im[mask] = val
                else:
                    im[mask] = np.maximum(im[mask], val)
            elif t == "hex":
                verts = self._hex_verts(el["center"][0], el["center"][1], el["radius"], el.get("rotation", 0.0))
                p = Path(verts)
                mask = self._rasterize_path(p, xg, yg)
                val = float(el.get("value", 1.0))
                if val < 1.0:
                    im[mask] = val
                else:
                    im[mask] = np.maximum(im[mask], val)
            elif t == "secondary":
                d = el["diameter"]
                mask = ((xg) ** 2 + (yg) ** 2) <= ((d / 2.0) ** 2)
                im[mask] = 0.0
            elif t == "spiders":
                arms = el["arms"]
                width = el["width"]
                angle0 = float(el.get("angle", 0.0))
                rmax = np.hypot(xg, yg).max()
                angles_list = el.get("angles", None)
                if angles_list is not None:
                    # explicit angles in degrees
                    use_angles = [np.deg2rad(a) for a in angles_list]
                else:
                    # evenly spaced arms around the circle
                    angle0_rad = np.deg2rad(angle0)
                    use_angles = [angle0_rad + 2 * np.pi * k / arms for k in range(arms)]
                # determine inner cutoff per-angle where spiders should start (e.g., secondary radius or central hex edge)
                inner_cuts = []
                # helper: ray-edge intersection for polygon centered at origin
                def ray_intersect_polygon(verts, dx, dy):
                    # verts: Nx2 array, polygon vertices in order
                    t_min = np.inf
                    for i in range(len(verts)):
                        p = verts[i]
                        q = verts[(i + 1) % len(verts)]
                        r = q - p
                        A = np.array([[dx, -r[0]], [dy, -r[1]]])
                        b = np.array([p[0], p[1]])
                        det = np.linalg.det(A)
                        if abs(det) < 1e-12:
                            continue
                        sol = np.linalg.solve(A, b)
                        t, u = sol[0], sol[1]
                        if t >= 0 and u >= 0 and u <= 1:
                            if t < t_min:
                                t_min = t
                    return t_min

                # precompute central occluder geometry if present
                central_hex_verts = None
                for e2 in self.elements:
                    if e2 is el:
                        continue
                    if e2.get("type") == "hex":
                        center2 = tuple(e2.get("center", (None, None)))
                        if center2 == (0.0, 0.0) and float(e2.get("value", 1.0)) < 1.0:
                            R = float(e2.get("radius", 0.0))
                            rot2 = float(e2.get("rotation", 0.0))
                            central_hex_verts = np.asarray(self._hex_verts(0.0, 0.0, R, rotation=rot2))
                    if e2.get("type") == "secondary":
                        # already handled below per-angle
                        pass

                for ang in use_angles:
                    # default inner cut from circular secondary if present
                    cut = 0.0
                    for e2 in self.elements:
                        if e2 is el:
                            continue
                        if e2.get("type") == "secondary":
                            cut = max(cut, float(e2.get("diameter", 0.0)) / 2.0)
                    # if there is a central hex occluder, compute ray intersection distance
                    dx, dy = np.cos(ang), np.sin(ang)
                    if central_hex_verts is not None:
                        t_hex = ray_intersect_polygon(central_hex_verts, dx, dy)
                        if np.isfinite(t_hex):
                            cut = max(cut, float(t_hex))
                    inner_cuts.append(float(cut))

                # now draw spiders using per-angle inner cut
                for ang, cut in zip(use_angles, inner_cuts):
                    xr = xg * np.cos(-ang) - yg * np.sin(-ang)
                    yr = xg * np.sin(-ang) + yg * np.cos(-ang)
                    mask = (np.abs(yr) <= (width / 2.0)) & (xr >= (cut - 1e-12))
                    im[mask] = 0.0
            elif t == "segments":
                # generate hex centers and vertices for a flat-top hex layout
                seg_flat = el["seg_flat"]
                rings = el["rings"]
                rot = el.get("rotation", 0.0)
                gap = el.get("gap", 0.0)
                # circumradius (center->vertex) from flat-to-flat: a = seg_flat / sqrt(3)
                a = seg_flat / np.sqrt(3.0)
                # drawn flat-to-flat reduced by gap, then compute draw circumradius
                drawn_flat = max(0.0, seg_flat - gap)
                a_draw = drawn_flat / np.sqrt(3.0)
                # generate centers using flat-top axial->cartesian conversion
                centers = []
                N = rings
                for q in range(-N, N + 1):
                    r1 = max(-N, -q - N)
                    r2 = min(N, -q + N)
                    for r in range(r1, r2 + 1):
                        x = a * 3.0 / 2.0 * q
                        y = a * np.sqrt(3.0) * (r + q / 2.0)
                        centers.append((x, y))
                # clip to primary diameter and draw hexes rotated so they are flat-top
                primR = self.diameter / 2.0
                for (cx, cy) in centers:
                    if np.hypot(cx, cy) <= primR + 1e-12:
                        verts = self._hex_verts(cx, cy, a_draw, rotation=rot)
                        p = Path(verts)
                        mask = self._rasterize_path(p, xg, yg)
                        im[mask] = 1.0
            else:
                # unknown element
                continue

        if ov > 1:
            # downsample by averaging blocks of size ov x ov
            H, W = im.shape
            im = im.reshape(H // ov, ov, W // ov, ov).mean(axis=(1, 3))

        # clip to [0,1]
        im = np.clip(im, 0.0, 1.0)
        return im

    def plot(self, npix: int = 512, soft: bool = True, oversample: int = 4, ax: Optional[_plt.Axes] = None, cmap: str = 'gray') -> _plt.Axes:
        """Plot the pupil and return the Matplotlib Axes."""
        arr = self.get_array(npix=npix, soft=soft, oversample=oversample)
        if ax is None:
            fig, ax = _plt.subplots()
        ax.imshow(arr, origin='lower', cmap=cmap, extent=[-self.diameter/2, self.diameter/2, -self.diameter/2, self.diameter/2])
        ax.set_xlabel('m')
        ax.set_ylabel('m')
        ax.set_aspect('equal')
        return ax

    # --- presets --------------------------------------------------------
    @staticmethod
    def jwst() -> 'Pupil':
        """Return a Pupil approximating JWST: 6.5 m primary, 18 segments, M2~0.74 m, 3 spiders."""
        p = Pupil(6.5 * u.m)
        # segmented primary only (no full filled disk)
        seg_flat = 1.2  # approximate flat-to-flat size per JWST segment (meters)
        rings = 2
        # small visible gap between segments
        p.add_segmented_primary(seg_flat=seg_flat, rings=rings, rotation=0.0, gap=0.02)
        # secondary: use a hexagon of the same flat-to-flat size as segments, blocking light
        # compute circumradius from flat-to-flat
        a = seg_flat / np.sqrt(3.0)
        p.add_hexagon(radius=a, center=(0.0, 0.0), value=0.0)
        # spiders: 3 branches with specified angles (degrees)
        # angles chosen so top branch is vertical (90 deg) and bottom two are 60 deg apart
        p.add_spiders(arms=3, width=0.06, angles=[90.0, 240.0, 300.0])
        return p

    @staticmethod
    def like(name: str) -> 'Pupil':
        """Return a preset pupil by common name (case-insensitive)."""
        key = (name or "").strip().lower()
        if key in ("jwst", "james webb", "james webb space telescope"):
            return Pupil.jwst()
        if key in ("vlt", "very large telescope"):
            return Pupil.vlt()
        if key in ("elt", "extremely large telescope"):
            return Pupil.elt()
        raise ValueError(f"Unknown pupil preset: {name}")

    @staticmethod
    def vlt() -> 'Pupil':
        """Return a Pupil approximating a single VLT Unit Telescope: 8.2 m primary, M2~1.1 m, 4 spiders."""
        p = Pupil(8.2 * u.m)
        p.add_disk(radius=8.2 / 2.0)
        p.add_central_obscuration(diameter=1.1)
        p.add_spiders(arms=4, width=0.05)
        return p

    @staticmethod
    def elt() -> 'Pupil':
        """Return a Pupil approximating ELT: 39.3 m primary segmented, M2~4.2 m, spiders approx 6."""
        p = Pupil(39.3 * u.m)
        p.add_disk(radius=39.3 / 2.0)
        # choose moderate segment size so many segments appear; this is only illustrative
        seg_flat = 1.45
        p.add_segmented_primary(seg_flat=seg_flat, rings=6)
        p.add_central_obscuration(diameter=4.2)
        p.add_spiders(arms=6, width=0.15)
        return p

class Collectors(Layer):
    """
    Represents the light collectors (telescopes).
    """
    def __init__(self, latitude: u.Quantity = 0*u.deg, longitude: u.Quantity = 0*u.deg, altitude: u.Quantity = 0*u.m, **kwargs):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.collectors = []
        super().__init__()

    def add(self, size: u.Quantity, shape: Pupil, position: Tuple[float, float], **kwargs):
        self.collectors.append({
            "size": size,
            "shape": shape,
            "position": position,
            **kwargs
        })

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Apply pupil mask to wavefront
        # Placeholder logic
        return wavefront

class BeamSplitter(Layer):
    def __init__(self, cutoff: float = 0.5):
        self.cutoff = cutoff
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> List[Wavefront]:
        # Split wavefront into two
        return [wavefront, wavefront] # Placeholder for copy

class Coronagraph(Layer):
    def __init__(self, phase_mask: str = '4quadrants'):
        self.phase_mask = phase_mask
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Apply coronagraphic mask
        return wavefront

class FiberIn(Layer):
    def __init__(self, modes: int = 1, **kwargs):
        self.modes = modes
        super().__init__()
    
    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Couple light into fiber
        return wavefront

class FiberOut(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        # Light exiting fiber
        return wavefront

def test_collectors():
    c = Collectors(latitude=0*u.deg, longitude=0*u.deg, altitude=0*u.m)
    p = Pupil(1*u.m)
    c.add(size=8*u.m, shape=p, position=(0,0))
    assert len(c.collectors) == 1

    # Test defaults
    default_c = Collectors()
    assert default_c.latitude == 0*u.deg
    assert default_c.longitude == 0*u.deg
    assert default_c.altitude == 0*u.m

if __name__ == "__main__":
    test_collectors()
    print("Optics tests passed.")
