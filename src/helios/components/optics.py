import numpy as np
import math
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
        # accept astropy.Quantity for radius
        if hasattr(radius, 'to'):
            r = float(radius.to(u.m).value)
        else:
            r = float(radius)
        self.elements.append({"type": "disk", "radius": r, "center": tuple(center), "value": float(value)})

    def add_hexagon(self, radius: float, center: Tuple[float, float] = (0.0, 0.0), value: float = 1.0, rotation: float = 0.0):
        """Add a regular hexagon (radius = circumradius)."""
        if hasattr(radius, 'to'):
            r = float(radius.to(u.m).value)
        else:
            r = float(radius)
        self.elements.append({"type": "hex", "radius": r, "center": tuple(center), "value": float(value), "rotation": float(rotation)})

    def add_central_obscuration(self, diameter: float):
        """Add central obscuration (diameter)."""
        if hasattr(diameter, 'to'):
            d = float(diameter.to(u.m).value)
        else:
            d = float(diameter)
        self.elements.append({"type": "secondary", "diameter": d})

    def add_spiders(self, arms: int = 4, width: float = 0.01, angle: float = 0.0, angles: Optional[List[float]] = None):
        """Add radially extended rectangular spider vanes.

        - `width` is in same linear units as `diameter`.
        - `angle` is rotation offset in degrees.
        """
        # allow width as Quantity
        if hasattr(width, 'to'):
            w = float(width.to(u.m).value)
        else:
            w = float(width)
        entry = {"type": "spiders", "arms": int(arms), "width": float(w), "angle": float(angle)}
        if angles is not None:
            # store explicit angles (degrees)
            entry["angles"] = [float(a) for a in angles]
        self.elements.append(entry)

    def add_segmented_primary(self, seg_flat: float, rings: int = 2, rotation: float = 0.0, gap: float = 0.0):
        """Create a hexagonal segmented primary with given flat-to-flat segment size.

        - `seg_flat`: flat-to-flat size of one segment (same units as `diameter`).
        - `rings`: number of hex rings around center (rings=0 -> 1 segment)
        """
        if hasattr(seg_flat, 'to'):
            sf = float(seg_flat.to(u.m).value)
        else:
            sf = float(seg_flat)
        if hasattr(gap, 'to'):
            g = float(gap.to(u.m).value)
        else:
            g = float(gap)
        self.elements.append({"type": "segments", "seg_flat": float(sf), "rings": int(rings), "rotation": float(rotation), "gap": float(g), "value": 1.0})

    # --- helpers rasterisation -----------------------------------------
    def _make_grid(self, npix: int, oversample: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
        size_m = self.diameter
        N = npix * oversample
        half = size_m / 2.0
        xs = np.linspace(-half, half, N, endpoint=False) + (size_m / N) / 2.0
        ys = xs.copy()
        xg, yg = np.meshgrid(xs, ys)
        return xg, yg, size_m / N

    def _hex_verts(self, cx: float, cy: float, R: float, rotation: float = 0.0):
        thetas = np.linspace(0, 2 * np.pi, 7) + np.deg2rad(rotation)
        return np.column_stack((cx + R * np.cos(thetas), cy + R * np.sin(thetas)))

    def _rasterize_path(self, path: Path, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
        pts = np.column_stack((xg.ravel(), yg.ravel()))
        mask = path.contains_points(pts)
        return mask.reshape(xg.shape)

    def get_array(self, npix: int = 256, soft: bool = False, oversample: int = 4) -> np.ndarray:
        """Retourne la pupille en tableau 2D (valeurs dans [0,1])."""
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
                im[mask] = np.maximum(im[mask], val) if val >= 1.0 else val
            elif t == "hex":
                verts = self._hex_verts(el["center"][0], el["center"][1], el["radius"], el.get("rotation", 0.0))
                pth = Path(verts)
                mask = self._rasterize_path(pth, xg, yg)
                val = float(el.get("value", 1.0))
                im[mask] = np.maximum(im[mask], val) if val >= 1.0 else val
            elif t == "secondary":
                d = el["diameter"]
                mask = (xg**2 + yg**2) <= (d/2.0)**2
                im[mask] = 0.0
            elif t == "spiders":
                arms = el["arms"]
                width = el["width"]
                angle0 = float(el.get("angle", 0.0))
                angles_list = el.get("angles", None)
                if angles_list is not None:
                    use_angles = [np.deg2rad(a) for a in angles_list]
                else:
                    base = np.deg2rad(angle0)
                    use_angles = [base + 2*np.pi*k/arms for k in range(arms)]
                # inner cutoff estimation (secondary radius if present)
                sec_rad = 0.0
                for e2 in self.elements:
                    if e2.get("type") == "secondary":
                        sec_rad = max(sec_rad, float(e2.get("diameter",0.0))/2.0)
                for ang in use_angles:
                    xr = xg * np.cos(-ang) - yg * np.sin(-ang)
                    yr = xg * np.sin(-ang) + yg * np.cos(-ang)
                    mask = (np.abs(yr) <= width/2.0) & (xr >= sec_rad - 1e-12)
                    im[mask] = 0.0
            elif t == "segments":
                seg_flat = el["seg_flat"]
                rings = el["rings"]
                rot = el.get("rotation", 0.0)
                gapv = el.get("gap", 0.0)
                val = float(el.get("value", 1.0))
                a = seg_flat / np.sqrt(3.0)
                drawn_flat = max(0.0, seg_flat - gapv)
                a_draw = drawn_flat / np.sqrt(3.0)
                centers = []
                N = rings
                for q in range(-N, N+1):
                    r1 = max(-N, -q - N)
                    r2 = min(N, -q + N)
                    for r in range(r1, r2+1):
                        cx = a * 1.5 * q
                        cy = a * np.sqrt(3.0) * (r + q/2.0)
                        centers.append((cx, cy))
                primR = self.diameter/2.0
                for (cx, cy) in centers:
                    if np.hypot(cx, cy) <= primR + 1e-12:
                        verts = self._hex_verts(cx, cy, a_draw, rotation=rot)
                        pth = Path(verts)
                        mask = self._rasterize_path(pth, xg, yg)
                        im[mask] = np.maximum(im[mask], val) if val >= 1.0 else val
            else:
                continue
        if ov > 1:
            H, W = im.shape
            im = im.reshape(H//ov, ov, W//ov, ov).mean(axis=(1,3))
        return np.clip(im, 0.0, 1.0)

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

    def diffraction_pattern(self, npix: int = 1024, soft: bool = True, oversample: int = 4, wavelength: float = 550e-9) -> np.ndarray:
        """Compute the (monochromatic) diffraction pattern (PSF intensity) of the pupil.

        - Returns a 2D NumPy array of shape (npix, npix) with values normalized to peak = 1.
        - This is a simple Fraunhofer-propagation via FFT of the pupil amplitude (pupil mask treated as amplitude transmission).
        - `soft` and `oversample` are forwarded to `get_array` to control anti-aliasing on the pupil.
        """
        # get pupil amplitude (transmission) array
        pup = self.get_array(npix=npix, soft=soft, oversample=oversample)
        # ensure float32 for fft speed
        pupf = np.asarray(pup, dtype=np.complex64)
        # compute FFT; use ifftshift/fftshift for correct centering
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupf)))
        intensity = np.abs(field) ** 2
        # normalize to peak 1
        maxv = intensity.max()
        if maxv > 0:
            intensity = intensity / float(maxv)
        return intensity

    def plot_diffraction_pattern(self, npix: int = 1024, soft: bool = True, oversample: int = 4, ax: Optional[_plt.Axes] = None, cmap: str = 'viridis', log: bool = True, vmax: Optional[float] = None, wavelength: float = 550e-9) -> _plt.Axes:
        """Plot the diffraction pattern (PSF) and return the Matplotlib Axes.

        - By default the intensity is shown on a log scale (dB-like) for dynamic range.
        - `vmax` can be used to clip the displayed maximum (after normalization).
        """
        intensity = self.diffraction_pattern(npix=npix, soft=soft, oversample=oversample, wavelength=wavelength)
        if ax is None:
            fig, ax = _plt.subplots()
        disp = intensity
        if log:
            # add tiny floor to avoid log(0)
            disp = np.log10(disp + 1e-12)
        # compute extent in units of lambda/D
        # frequency axis fx = (i - N/2) / (N * dx) cycles/m
        # units lambda/D = fx * D
        N = intensity.shape[0]
        # dx used when building pupil: size / N_pixels where size = self.diameter
        dx = self.diameter / float(N)
        fx = (np.arange(N) - N // 2) / (N * dx)
        # lambda/D units
        lamD = fx * self.diameter
        extent = [lamD[0], lamD[-1], lamD[0], lamD[-1]]
        im = ax.imshow(disp, origin='lower', cmap=cmap, extent=extent)
        ax.set_xlabel('Focal plane (arb. units)')
        ax.set_ylabel('Focal plane (arb. units)')
        ax.set_aspect('equal')
        if vmax is not None:
            im.set_clim(vmin=disp.min(), vmax=vmax)
        _plt.colorbar(im, ax=ax)
        return ax

    def image_through_pupil(self, scene_array: np.ndarray, soft: bool = True, oversample: int = 4, normalize: bool = True) -> np.ndarray:
        """Compute the image formed by the optical system when the given `scene_array` is observed through this pupil.

        Assumptions / algorithm (simple Fraunhofer imaging, monochromatic):
        - `scene_array` is a 2D real array representing the object intensity in object plane (image of object at infinite distance).
        - We compute the Fourier transform of the object, multiply by the pupil amplitude (transmission), then inverse FT to obtain the complex image field. The output is the intensity (|field|^2).

        Returns a 2D NumPy array (float) with same shape as `scene_array`.
        """
        # validate input
        arr = np.asarray(scene_array, dtype=np.complex64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("scene_array must be a square 2D array")
        N = arr.shape[0]
        # get pupil amplitude at same sampling
        pup = self.get_array(npix=N, soft=soft, oversample=oversample)
        pup_amp = np.asarray(pup, dtype=np.complex64)

        # object -> pupil plane (FT)
        obj_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
        # apply pupil (amplitude transmission)
        field_after = obj_field * pup_amp
        # back to image plane (inverse FT)
        img_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_after)))
        intensity = np.abs(img_field) ** 2
        if normalize:
            m = intensity.max()
            if m > 0:
                intensity = intensity / float(m)
        return intensity

    def plot_image_through_pupil(self, scene_array: np.ndarray, soft: bool = True, oversample: int = 4, ax: Optional[_plt.Axes] = None, cmap: str = 'gray', normalize: bool = True, log: bool = False) -> _plt.Axes:
        """Compute and plot the image of `scene_array` formed through this pupil. Returns the Matplotlib Axes."""
        intensity = self.image_through_pupil(scene_array, soft=soft, oversample=oversample, normalize=normalize)
        if ax is None:
            fig, ax = _plt.subplots()
        disp = intensity
        if log:
            disp = np.log10(disp + 1e-12)
        im = ax.imshow(disp, origin='lower', cmap=cmap)
        ax.set_xlabel('Image x (pixels)')
        ax.set_ylabel('Image y (pixels)')
        ax.set_aspect('equal')
        _plt.colorbar(im, ax=ax)
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
        """Construire une pupille ELT selon les nouvelles règles:

        - Taille coin-à-coin d'un segment (primaire & secondaire): 1.45 m (circumradius = 0.725 m)
        - Secondaire: 4 anneaux hexagonaux (total 61) tous occultants (value=0)
        - Primaire: exactement 798 segments transmissifs formant un dodécagone minimal
          obtenu par recherche binaire sur l'apothème puis éventuel trimming intérieur.
        - Araignées: 6 bras conservés (angles définis).
        """
        p = Pupil(39.3 * u.m)

        corner_to_corner = 1.45
        R_circum = corner_to_corner / 2.0
        a_draw = R_circum
        a = R_circum

        def generate_centers(rings: int) -> List[Tuple[float, float]]:
            centers = []
            N = rings
            for q in range(-N, N + 1):
                r1 = max(-N, -q - N)
                r2 = min(N, -q + N)
                for r in range(r1, r2 + 1):
                    cx = a * 1.5 * q
                    cy = a * np.sqrt(3.0) * (r + q / 2.0)
                    centers.append((cx, cy))
            return centers

        # (Ancien) test d'inclusion complète non utilisé désormais.
        def center_inside(cx: float, cy: float, poly: Path) -> bool:
            return poly.contains_point((cx, cy))

        rings_primary = 40
        lattice = generate_centers(rings_primary)

        n_sides = 12
        angle_step = 2.0 * np.pi / n_sides
        max_center_radius = max(np.hypot(cx, cy) for cx, cy in lattice)
        lower_ap = (4 + 1) * 1.5 * a * 0.5
        upper_ap = max_center_radius + a_draw

        best_sel = None
        best_ap = None

        # Rotation choisie pour avoir deux côtés horizontaux (haut & bas): rot = 5π/12
        rot = 5.0 * np.pi / 12.0

        def build_polygon(ap):
            circum = ap / np.cos(np.pi / n_sides)
            return [(circum * np.cos(rot + k * angle_step), circum * np.sin(rot + k * angle_step)) for k in range(n_sides)]

        def center_inside_tol(cx, cy, verts, tol=1e-9):
            vp = np.asarray(verts, dtype=float)
            edges = list(zip(vp, np.roll(vp, -1, axis=0)))
            dmins = []
            for (p1, p2) in edges:
                ex, ey = p2 - p1
                nx, ny = -ey, ex
                L = np.hypot(nx, ny)
                if L == 0:
                    continue
                nx /= L; ny /= L
                d = (cx - p1[0]) * nx + (cy - p1[1]) * ny
                dmins.append(d)
            return (len(dmins) > 0) and (min(dmins) >= -tol)

        # Recherche binaire pour obtenir 859 hexagones totaux (798 primaire + 61 secondaire)
        target_total = 859
        for _ in range(60):
            mid = 0.5 * (lower_ap + upper_ap)
            poly_verts = build_polygon(mid)
            inside = [c for c in lattice if center_inside_tol(c[0], c[1], poly_verts)]
            if len(inside) >= target_total:
                best_sel = inside
                best_ap = mid
                upper_ap = mid
            else:
                lower_ap = mid

        if best_sel is None:
            raise RuntimeError(f"Impossible de trouver un dodécagone contenant >={target_total} segments.")

        # Appliquer facteur d'échelle 1.03 pour ajustement fin
        best_ap = best_ap * 1.03
        poly_verts_final = build_polygon(best_ap)
        best_sel = [c for c in lattice if center_inside_tol(c[0], c[1], poly_verts_final)]

        if len(best_sel) > 798:
            # Sélection par groupes de symétrie (x,y) -> (±x, ±y)
            target = 798
            # Regrouper par valeurs absolues pour identifier les familles de symétrie
            groups = {}
            for (cx, cy) in best_sel:
                key = (round(abs(cx), 6), round(abs(cy), 6))
                groups.setdefault(key, []).append((cx, cy))
            # Ordonner les groupes du bord vers le centre par rayon
            ordered_keys = sorted(groups.keys(), key=lambda k: k[0]*k[0] + k[1]*k[1], reverse=True)
            selected = []
            remaining = target
            for key in ordered_keys:
                pts = groups[key]
                # Si groupe passe entièrement
                if len(pts) <= remaining:
                    selected.extend(pts)
                    remaining -= len(pts)
                    if remaining == 0:
                        break
                else:
                    # Sélection partielle dans un groupe trop grand : essayer de prendre des paires symétriques
                    # Indexation par signature de signes
                    sign_map = {}
                    for (cx, cy) in pts:
                        sign_map.setdefault((int(np.sign(cx)), int(np.sign(cy))), []).append((cx, cy))
                    chosen = []
                    # Chercher d'abord paires miroir horizontal (x et -x)
                    for (sx, sy), lst in list(sign_map.items()):
                        opp = (-sx, sy)
                        if opp in sign_map and sx != 0:  # paire horizontale
                            chosen.append(lst[0]); chosen.append(sign_map[opp][0])
                            if len(chosen) >= remaining:
                                break
                    # Puis paires miroir vertical (y et -y) si nécessaire
                    if len(chosen) < remaining:
                        for (sx, sy), lst in list(sign_map.items()):
                            opp = (sx, -sy)
                            if opp in sign_map and sy != 0:
                                # éviter doublons déjà pris
                                cand1, cand2 = lst[0], sign_map[opp][0]
                                if cand1 not in chosen and cand2 not in chosen:
                                    chosen.append(cand1); chosen.append(cand2)
                                    if len(chosen) >= remaining:
                                        break
                    # Si encore insuffisant, compléter arbitrairement (mais toujours symétrique si possible)
                    if len(chosen) < remaining:
                        for (cx, cy) in pts:
                            if (cx, cy) not in chosen:
                                chosen.append((cx, cy))
                                if len(chosen) >= remaining:
                                    break
                    selected.extend(chosen[:remaining])
                    remaining = 0
                    break
            # Si pour une raison quelconque il reste des places (rare), compléter par rayon décroissant restant
            if remaining > 0:
                leftover = [c for c in best_sel if c not in selected]
                leftover_sorted = sorted(leftover, key=lambda c: c[0]*c[0] + c[1]*c[1], reverse=True)
                selected.extend(leftover_sorted[:remaining])
            primary_final = selected
        else:
            primary_final = best_sel

        for (cx, cy) in primary_final:
            p.add_hexagon(radius=a_draw, center=(cx, cy), value=1.0)

        central_rings = 4
        for (cx, cy) in generate_centers(central_rings):
            p.add_hexagon(radius=a_draw, center=(cx, cy), value=0.0)

        angles = [90.0, 270.0, 30.0, 150.0, 210.0, 330.0]
        p.add_spiders(arms=6, width=0.25, angles=angles)

        return p

class Collectors(Layer):
    """Represents the light collectors (telescopes) with optional pupil geometry.
    
    Each collector can be associated with a specific pupil (aperture geometry) that 
    defines its transmission pattern. The pupil is applied to the wavefront during 
    processing.
    
    Parameters
    ----------
    latitude : astropy.Quantity
        Geographic latitude of the observatory (degrees).
    longitude : astropy.Quantity
        Geographic longitude of the observatory (degrees).
    altitude : astropy.Quantity
        Altitude of the observatory above sea level (meters).
    
    Examples
    --------
    >>> collectors = Collectors(latitude=24.6*u.deg, altitude=2400*u.m)
    >>> pupil = Pupil.like('VLT')
    >>> collectors.add(size=8*u.m, pupil=pupil, position=(0, 0))
    """
    def __init__(self, latitude: u.Quantity = 0*u.deg, longitude: u.Quantity = 0*u.deg, altitude: u.Quantity = 0*u.m, **kwargs):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.collectors = []
        super().__init__()

    def add(self, size: u.Quantity, pupil: Optional[Pupil] = None, position: Tuple[float, float] = (0, 0), **kwargs):
        """Add a collector to this observatory.
        
        Parameters
        ----------
        size : astropy.Quantity
            Diameter of the collector aperture (meters).
        pupil : Pupil, optional
            Pupil geometry defining the aperture transmission. If None, a simple
            circular aperture is assumed.
        position : Tuple[float, float]
            (x, y) position of the collector in the aperture plane (meters).
            For single telescopes, use (0, 0). For interferometers, specify
            baseline coordinates.
        """
        self.collectors.append({
            "size": size,
            "pupil": pupil,
            "shape": pupil,  # backward compatibility
            "position": position,
            **kwargs
        })

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        """Apply pupil mask to wavefront.
        
        For each configured collector, if a pupil geometry is provided,
        rasterize it to the wavefront sampling and multiply the complex
        field by the pupil amplitude (transmission).
        
        Multiple collectors are combined multiplicatively (for co-phased arrays)
        or as separate apertures (for interferometers).
        """
        try:
            N = wavefront.field.shape[0]
        except Exception:
            return wavefront

        total_mask = np.ones((N, N), dtype=float)
        for col in self.collectors:
            # Try new 'pupil' key first, fallback to 'shape' for backward compatibility
            pupil = col.get("pupil", col.get("shape", None))
            if isinstance(pupil, Pupil):
                # Assume the pupil was created with the collector diameter in meters
                # If not, users should construct the Pupil with the collector size.
                try:
                    mask = pupil.get_array(npix=N, soft=True)
                except Exception:
                    # fallback: try without anti-aliasing
                    mask = pupil.get_array(npix=N, soft=False)
                # combine masks multiplicatively (multiple collectors/telescopes)
                total_mask = total_mask * mask

        # apply mask to complex field (amplitude transmission)
        wavefront.field = wavefront.field * total_mask.astype(wavefront.field.dtype)
        return wavefront


class Interferometer(Layer):
    """Interferometric array of multiple collectors with individual pupil geometries.
    
    An interferometer combines light from multiple spatially separated collectors.
    Each collector can have its own pupil geometry and position in the (u,v) plane
    (baseline coordinates).
    
    This class is useful for modeling:
    - Optical interferometers (VLTI, CHARA, etc.)
    - Aperture masking interferometry
    - Dilute aperture arrays
    
    Parameters
    ----------
    name : str, optional
        Name of the interferometer configuration (e.g., "VLTI-UT", "CHARA").
    
    Attributes
    ----------
    collectors : list
        List of individual collectors, each defined as a dictionary with keys:
        - 'pupil': Pupil geometry
        - 'position': (x, y) baseline coordinates in meters
        - 'size': aperture diameter
    
    Examples
    --------
    >>> # Create a simple 2-telescope interferometer
    >>> vlti = Interferometer(name="VLTI-UT")
    >>> pupil_ut = Pupil.like('VLT')
    >>> vlti.add_collector(pupil=pupil_ut, position=(0, 0), size=8.2*u.m)
    >>> vlti.add_collector(pupil=pupil_ut, position=(47, 0), size=8.2*u.m)
    >>> 
    >>> # 3-telescope configuration
    >>> chara = Interferometer(name="CHARA")
    >>> for pos in [(0, 0), (100, 0), (50, 86.6)]:
    >>>     chara.add_collector(pupil=Pupil(1*u.m), position=pos, size=1*u.m)
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or "Interferometer"
        self.collectors = []
    
    def add_collector(self, pupil: Pupil, position: Tuple[float, float], 
                     size: Optional[u.Quantity] = None, **kwargs):
        """Add a collector to the interferometric array.
        
        Parameters
        ----------
        pupil : Pupil
            Pupil geometry for this collector (defines aperture shape).
        position : Tuple[float, float]
            (x, y) baseline coordinates in meters. This defines the spatial
            separation between collectors in the aperture plane.
        size : astropy.Quantity, optional
            Diameter of the collector. If None, inferred from pupil.diameter.
        **kwargs
            Additional metadata (e.g., telescope name, mount type).
        """
        if size is None:
            # Try to infer from pupil
            if hasattr(pupil, 'diameter'):
                size = pupil.diameter * u.m
            else:
                size = 1.0 * u.m
        
        self.collectors.append({
            "pupil": pupil,
            "position": tuple(position),
            "size": size,
            **kwargs
        })
    
    def get_baseline_array(self) -> np.ndarray:
        """Return array of baseline vectors (u,v coordinates) in meters.
        
        Returns
        -------
        baselines : ndarray
            Array of shape (N, 2) where N is the number of collectors.
            Each row is (x, y) position in meters.
        """
        return np.array([c["position"] for c in self.collectors], dtype=float)
    
    def plot_array(self, ax: Optional[_plt.Axes] = None, show_pupils: bool = True,
                  pupil_scale: float = 1.0) -> _plt.Axes:
        """Plot the interferometer array configuration.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_pupils : bool
            If True, render individual pupil shapes at each baseline position.
        pupil_scale : float
            Scale factor for pupil rendering (1.0 = actual size).
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            fig, ax = _plt.subplots(figsize=(8, 8))
        
        baselines = self.get_baseline_array()
        
        if show_pupils and len(self.collectors) > 0:
            # Determine plot extent
            max_extent = 0
            for c in self.collectors:
                x, y = c["position"]
                size = c["size"].to(u.m).value if hasattr(c["size"], 'to') else float(c["size"])
                max_extent = max(max_extent, abs(x) + size, abs(y) + size)
            
            # Render each pupil at its position
            for c in self.collectors:
                pupil = c["pupil"]
                x, y = c["position"]
                
                # Render pupil at moderate resolution
                npix_pupil = 128
                pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
                
                # Determine physical extent of this pupil
                diam = pupil.diameter * pupil_scale
                extent_pupil = [x - diam/2, x + diam/2, y - diam/2, y + diam/2]
                
                ax.imshow(pupil_arr, origin='lower', cmap='gray', alpha=0.7, extent=extent_pupil)
        else:
            # Simple scatter plot
            ax.scatter(baselines[:, 0], baselines[:, 1], s=100, c='blue', 
                      marker='o', edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Baseline x (m)')
        ax.set_ylabel('Baseline y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{self.name} - Array Configuration ({len(self.collectors)} collectors)')
        
        return ax
    
    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        """Apply interferometric array to wavefront.
        
        This combines the apertures of all collectors. For true interferometric
        imaging (fringe formation), use dedicated photonics/beam combination layers.
        
        Here we simply combine all pupil masks in the aperture plane.
        """
        try:
            N = wavefront.field.shape[0]
        except Exception:
            return wavefront
        
        # Build combined aperture mask
        combined_mask = np.zeros((N, N), dtype=float)
        
        # Physical extent of the wavefront array (assume it covers the full baseline)
        baselines = self.get_baseline_array()
        if len(baselines) == 0:
            return wavefront
        
        # Determine array extent
        max_extent = np.max(np.abs(baselines)) if len(baselines) > 0 else 1.0
        for c in self.collectors:
            size_m = c["size"].to(u.m).value if hasattr(c["size"], 'to') else float(c["size"])
            max_extent = max(max_extent, size_m)
        
        # Pixel scale: meters per pixel
        pixel_scale = 2.0 * max_extent / float(N)
        
        # Render each collector pupil at its position
        for c in self.collectors:
            pupil = c["pupil"]
            x, y = c["position"]
            
            # Render pupil
            pupil_arr = pupil.get_array(npix=N, soft=True)
            
            # Compute shift in pixels
            shift_x = int(x / pixel_scale)
            shift_y = int(y / pixel_scale)
            
            # Shift pupil to baseline position
            shifted_pupil = np.roll(pupil_arr, shift=(shift_y, shift_x), axis=(0, 1))
            
            # Add to combined mask
            combined_mask = np.maximum(combined_mask, shifted_pupil)
        
        # Apply to wavefront
        wavefront.field = wavefront.field * combined_mask.astype(wavefront.field.dtype)
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
        # Apply a simple focal-plane coronagraph mask to the incoming wavefront.
        # Algorithm (Fraunhofer approximation, monochromatic):
        # 1. FFT(wavefront.field) -> focal plane field
        # 2. Multiply by focal-plane mask (phase and/or amplitude)
        # 3. inverse FFT -> back to pupil/image plane
        try:
            field = wavefront.field
            N = field.shape[0]
        except Exception:
            return wavefront

        # focal-plane field
        ffield = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

        # build mask
        mask = self.mask_array(npix=N)

        # apply mask in focal plane
        ffield_masked = ffield * mask

        # back to pupil/image plane
        field_after = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ffield_masked)))
        wavefront.field = field_after.astype(wavefront.field.dtype)
        return wavefront

    def mask_array(self, npix: int = 512, kind: str = None, charge: int = 2,
                   lam: Optional[u.Quantity] = None,
                   diameter: Optional[u.Quantity] = None,
                   fov: Optional[u.Quantity] = None) -> np.ndarray:
        """Return a complex focal-plane mask array (shape npix x npix).

        - `kind`: overrides self.phase_mask if provided.
        - `lam` and `diameter` (both astropy Quantities) can be provided to express
          coordinates in units of lambda/D (angular units). In that case `fov`
          (angular total field-of-view, e.g. `4*u.arcsec`) should also be provided
          so the mask pixels can be mapped to angular coordinates.
        - Supported masks: '4quadrants' (pi phase shifts), 'vortex' (charge >=1,
          phase exp(i*charge*theta)). If no `lam`/`diameter` are given the mask
          is generated on a normalized [-1,1] grid as before.
        """
        k = kind or self.phase_mask or '4quadrants'

        # Build coordinate grid. If physical angular scaling provided, map pixels
        # to angular coordinates and then to units of lambda/D; otherwise use
        # a normalized [-1,1] grid as legacy behavior.
        if (lam is not None) and (diameter is not None) and (fov is not None):
            # ensure astropy quantities
            lam = lam.to(u.m)
            diameter = diameter.to(u.m)
            # fov is expected to be an angle (e.g., arcsec)
            fov_angle = fov.to(u.rad).value
            # angular resolution per pixel (radians)
            xs = np.linspace(-fov_angle / 2.0, fov_angle / 2.0, npix)
            ys = xs.copy()
            xg_ang, yg_ang = np.meshgrid(xs, ys)
            # lambda/D in radians
            lam_over_D = (lam / diameter).decompose().value
            # coordinates in units of lambda/D
            xg = xg_ang / lam_over_D
            yg = yg_ang / lam_over_D
        else:
            xs = np.linspace(-1.0, 1.0, npix)
            ys = xs.copy()
            xg, yg = np.meshgrid(xs, ys)

        if k.lower() in ('4quadrants', '4q'):
            # 4QPM: alternate quadrants have pi phase (i.e., -1 complex multiplier)
            mask = np.ones((npix, npix), dtype=np.complex64)
            mask[(xg < 0) & (yg > 0)] = -1.0
            mask[(xg > 0) & (yg < 0)] = -1.0
            return mask
        elif k.lower() in ('vortex',):
            theta = np.arctan2(yg, xg)
            phase = np.exp(1j * charge * theta)
            return phase.astype(np.complex64)
        else:
            # identity (no mask)
            return np.ones((npix, npix), dtype=np.complex64)

    def plot_mask(self, npix: int = 512, kind: str = None, charge: int = 2,
                  lam: Optional[u.Quantity] = None,
                  diameter: Optional[u.Quantity] = None,
                  fov: Optional[u.Quantity] = None,
                  ax: Optional[_plt.Axes] = None,
                  cmap: str = 'gray',
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None,
                  display: Optional[str] = None) -> _plt.Axes:
        """Plot the coronagraph focal-plane mask phase (in radians) as a grayscale image.

        - The mask is built via `mask_array(...)`. The plotted quantity is the
          wrapped phase angle `np.angle(mask)` in radians. For purely real masks
          (e.g. 4-quadrant), values will be 0 or pi.
        - Parameters are forwarded to `mask_array` to allow physical scaling.
        - Returns the Matplotlib Axes containing the image.
        """
        mask = self.mask_array(npix=npix, kind=kind, charge=charge, lam=lam, diameter=diameter, fov=fov)
        # compute phase in range [-pi, pi]
        phase = np.angle(mask)
        if ax is None:
            fig, ax = _plt.subplots()
        # Determine extent (so image axes are in physical units centered on 0,0)
        extent = None
        xlabel = 'Focal plane x (pixels)'
        ylabel = 'Focal plane y (pixels)'
        if (lam is not None) and (diameter is not None) and (fov is not None):
            # physical scaling available
            # default display is lambda/D
            display_mode = (display or 'lambda/D').lower()
            # fov in radians
            fov_rad = float(fov.to(u.rad).value)
            lam_over_D = float((lam / diameter).to(u.dimensionless_unscaled).value)
            if display_mode in ('lambda/d', 'lambda/d', 'lambdad', 'lambda/d)') or display_mode == 'lambda/d':
                # extent in units of lambda/D
                half = (fov_rad / 2.0) / lam_over_D
                extent = [-half, half, -half, half]
                xlabel = 'Focal plane x (lambda/D)'
                ylabel = 'Focal plane y (lambda/D)'
            elif display_mode in ('arcsec', 'arcseconds'):
                half = float(fov.to(u.arcsec).value) / 2.0
                extent = [-half, half, -half, half]
                xlabel = 'Focal plane x (arcsec)'
                ylabel = 'Focal plane y (arcsec)'
            elif display_mode in ('rad', 'radians'):
                half = fov_rad / 2.0
                extent = [-half, half, -half, half]
                xlabel = 'Focal plane x (rad)'
                ylabel = 'Focal plane y (rad)'
            else:
                # fallback to pixels if unknown
                extent = [-npix/2.0, npix/2.0, -npix/2.0, npix/2.0]
        else:
            extent = [-npix/2.0, npix/2.0, -npix/2.0, npix/2.0]

        im = ax.imshow(phase, origin='lower', cmap=cmap, extent=extent)
        if vmin is not None or vmax is not None:
            im.set_clim(vmin=vmin if vmin is not None else phase.min(), vmax=vmax if vmax is not None else phase.max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        _plt.colorbar(im, ax=ax, label='Phase (rad)')
        return ax

    def image_from_scene(self, scene_array: np.ndarray, soft: bool = True, oversample: int = 4, normalize: bool = True,
                         lam: Optional[u.Quantity] = None, diameter: Optional[u.Quantity] = None, fov: Optional[u.Quantity] = None) -> np.ndarray:
        """Compute the image obtained after placing this coronagraph in the focal plane.

        Simplified pipeline (monochromatic, Fraunhofer):
        - FFT(scene_array) -> focal plane field
        - apply coronagraph focal-plane mask
        - inverse FFT -> image field, return intensity
        """
        arr = np.asarray(scene_array, dtype=np.complex64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError('scene_array must be a square 2D array')
        N = arr.shape[0]
        # field in focal plane
        ffield = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
        # build mask, forwarding physical scaling if provided
        mask = self.mask_array(npix=N, lam=lam, diameter=diameter, fov=fov)
        ffield_masked = ffield * mask
        img_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ffield_masked)))
        intensity = np.abs(img_field) ** 2
        if normalize:
            m = intensity.max()
            if m > 0:
                intensity = intensity / float(m)
        return intensity

    def plot_image_from_scene(self, scene_array: np.ndarray, ax: Optional[_plt.Axes] = None, cmap: str = 'inferno', normalize: bool = True, log: bool = False) -> _plt.Axes:
        """Compute and plot the image after coronagraphic mask. Returns Matplotlib Axes."""
        intensity = self.image_from_scene(scene_array, normalize=normalize)
        disp = intensity
        if log:
            disp = np.log10(disp + 1e-12)
        if ax is None:
            fig, ax = _plt.subplots()
        im = ax.imshow(disp, origin='lower', cmap=cmap)
        ax.set_xlabel('Image x (pixels)')
        ax.set_ylabel('Image y (pixels)')
        ax.set_aspect('equal')
        _plt.colorbar(im, ax=ax)
        return ax

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


class Atmosphere(Layer):
    """Kolmogorov atmosphere layer producing chromatic phase screens with frozen-flow turbulence.

    The atmosphere introduces optical path difference (OPD) errors that are chromatic - 
    the phase shift depends on wavelength: phi = 2π * OPD / λ.

    Temporal evolution is modeled via frozen-flow (Taylor hypothesis): turbulent screens 
    drift at constant wind velocity, and different observation times sample different 
    regions of the frozen turbulent volume.

    Parameters
    ----------
    rms : astropy.Quantity
        Desired RMS of the OPD (optical path difference) in length units (e.g., nm, μm).
        This is the wavefront error amplitude, NOT phase in radians.
        Default: 100 nm (good seeing conditions at visible wavelengths).
    
    wind_speed : astropy.Quantity
        Wind velocity vector magnitude and direction. Can be:
        - Scalar Quantity: wind speed in m/s (default direction: +x)
        - Tuple of 2 Quantities: (vx, vy) wind velocity components in m/s
        Default: 5 m/s in +x direction (~18 km/h, typical high-altitude wind).
    
    wind_direction : float, optional
        Wind direction in degrees (0° = +x, 90° = +y). Used only if wind_speed is scalar.
        Default: 0°.
    
    seed : int, optional
        RNG seed for reproducible turbulent realizations. If None, uses random seed.
        The same seed produces the same frozen turbulent volume.
        Default: None (random).
    
    inner_scale : astropy.Quantity, optional
        Inner scale of turbulence (l0) in meters. Below this scale, turbulence becomes 
        isotropic. Default: None (pure Kolmogorov, no inner scale).
    
    outer_scale : astropy.Quantity, optional
        Outer scale of turbulence (L0) in meters. Above this scale, turbulence energy 
        saturates. Default: None (infinite outer scale).
    
    Notes
    -----
    - The phase screen is generated in Fourier space using Kolmogorov statistics (f^-11/3 PSD).
    - Time evolution: screen drifts at wind_speed, so screen(t) = screen(x - v*t, y).
    - Chromatic behavior: phase(λ) = 2π * OPD / λ, so shorter wavelengths see larger phase.
    
    Examples
    --------
    >>> # Good seeing: 100nm RMS OPD, 5 m/s wind
    >>> atm = Atmosphere(rms=100*u.nm, wind_speed=5*u.m/u.s, seed=42)
    >>> 
    >>> # Poor seeing: 500nm RMS OPD, strong wind
    >>> atm = Atmosphere(rms=500*u.nm, wind_speed=15*u.m/u.s)
    >>> 
    >>> # Custom wind direction (northeast)
    >>> atm = Atmosphere(rms=200*u.nm, wind_speed=10*u.m/u.s, wind_direction=45)
    """
    def __init__(self, 
                 rms: u.Quantity = 100*u.nm,
                 wind_speed: Union[u.Quantity, Tuple[u.Quantity, u.Quantity]] = 5*u.m/u.s,
                 wind_direction: float = 0.0,
                 seed: Optional[int] = None,
                 inner_scale: Optional[u.Quantity] = None,
                 outer_scale: Optional[u.Quantity] = None):
        super().__init__()
        
        # Store OPD RMS in meters
        if hasattr(rms, 'to'):
            self.rms = float(rms.to(u.m).value)
        else:
            # If no unit, assume meters
            self.rms = float(rms)
        
        # Parse wind velocity
        if isinstance(wind_speed, tuple):
            # (vx, vy) components provided
            vx = wind_speed[0].to(u.m/u.s).value if hasattr(wind_speed[0], 'to') else float(wind_speed[0])
            vy = wind_speed[1].to(u.m/u.s).value if hasattr(wind_speed[1], 'to') else float(wind_speed[1])
            self.wind_velocity = np.array([vx, vy], dtype=float)
        else:
            # Scalar speed + direction
            speed = wind_speed.to(u.m/u.s).value if hasattr(wind_speed, 'to') else float(wind_speed)
            angle_rad = np.deg2rad(wind_direction)
            self.wind_velocity = np.array([speed * np.cos(angle_rad), 
                                          speed * np.sin(angle_rad)], dtype=float)
        
        # Seed for reproducibility
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        
        # Turbulence scales in meters (optional, for Von Karman model)
        self.inner_scale = None
        self.outer_scale = None
        if inner_scale is not None:
            if hasattr(inner_scale, 'to'):
                self.inner_scale = float(inner_scale.to(u.m).value)
            else:
                self.inner_scale = float(inner_scale)
        if outer_scale is not None:
            if hasattr(outer_scale, 'to'):
                self.outer_scale = float(outer_scale.to(u.m).value)
            else:
                self.outer_scale = float(outer_scale)
        
        # Cache for frozen turbulent screen
        self._frozen_screen = None
        self._screen_size = None


    def _generate_frozen_screen(self, N: int, oversample: int = 2) -> np.ndarray:
        """Generate a large frozen turbulent screen for temporal evolution.
        
        The screen is oversampled to allow smooth translation without aliasing.
        
        Parameters
        ----------
        N : int
            Base array size (will be multiplied by oversample)
        oversample : int
            Oversampling factor for smooth translation (default: 2)
        
        Returns
        -------
        screen : ndarray
            OPD screen in meters, shape (N*oversample, N*oversample)
        """
        Nlarge = N * oversample
        rng = np.random.default_rng(self.seed)
        
        # Generate frequency grid (cycles per pixel)
        fx = np.fft.fftfreq(Nlarge)
        fy = fx.copy()
        fxg, fyg = np.meshgrid(fx, fy)
        f = np.sqrt(fxg ** 2 + fyg ** 2)

        # Avoid zero frequency singularity
        nonzero = f[f > 0]
        if nonzero.size == 0:
            fmin = 1.0 / float(Nlarge)
        else:
            fmin = float(nonzero.min())
        f[0, 0] = fmin

        # Kolmogorov filter amplitude ~ f^{-11/6} (sqrt of PSD ~ f^{-11/3})
        with np.errstate(divide='ignore', invalid='ignore'):
            filt = f ** (-11.0 / 6.0)
        
        # Apply Von Karman modifications if scales are specified
        if self.inner_scale is not None or self.outer_scale is not None:
            # Convert frequency to spatial scale (in pixels)
            # For inner scale cutoff: high-pass filter
            if self.inner_scale is not None:
                # inner scale in pixels (assuming screen spans self.diameter)
                # This is approximate - proper implementation needs pupil diameter info
                pass  # TODO: implement Von Karman inner scale
            
            if self.outer_scale is not None:
                # outer scale cutoff: low-pass filter
                pass  # TODO: implement Von Karman outer scale
        
        # Cap extreme values
        filt = np.nan_to_num(filt, nan=filt.max(), posinf=filt.max(), neginf=0.0)

        # Generate complex Gaussian white noise in Fourier domain
        real = rng.normal(size=(Nlarge, Nlarge))
        imag = rng.normal(size=(Nlarge, Nlarge))
        fourier = (real + 1j * imag) * filt

        # Zero DC component
        fourier[0, 0] = 0.0 + 0.0j

        # Enforce Hermitian symmetry for real-valued output
        fourier = (fourier + np.conj(np.flipud(np.fliplr(fourier)))) / 2.0

        # Inverse FFT to get OPD screen (real-valued, in arbitrary units)
        opd_screen = np.fft.ifft2(fourier).real

        # Normalize to requested RMS (in meters)
        screen_rms = float(np.std(opd_screen))
        if screen_rms <= 0 or not np.isfinite(screen_rms):
            return np.zeros((Nlarge, Nlarge), dtype=float)
        
        opd_screen = opd_screen / screen_rms * float(self.rms)
        return opd_screen

    def _extract_screen_at_time(self, time: u.Quantity, N: int) -> np.ndarray:
        """Extract N×N screen from frozen turbulence at given time.
        
        Uses bilinear interpolation to extract a shifted window from the frozen screen.
        
        Parameters
        ----------
        time : astropy.Quantity
            Observation time (shift = wind_velocity * time)
        N : int
            Output array size
        
        Returns
        -------
        screen : ndarray
            OPD screen in meters, shape (N, N)
        """
        # Ensure frozen screen exists
        if self._frozen_screen is None or self._screen_size != N:
            self._frozen_screen = self._generate_frozen_screen(N, oversample=2)
            self._screen_size = N
        
        # Convert time to shift in pixels
        if hasattr(time, 'to'):
            time_s = time.to(u.s).value
        else:
            time_s = float(time)
        
        # Compute shift in meters: displacement = velocity * time
        # Assume screen pixel size ~ diameter / N (approximate)
        # For now, use normalized shift (shift in units of N)
        # More robust: shift_pixels = (wind_velocity * time) / pixel_physical_size
        # Simple approach: shift in fraction of array size
        Nlarge = self._frozen_screen.shape[0]
        
        # Shift in pixels (assume screen spans ~2*diameter to allow drift)
        # Pixel size in frozen screen: diameter / N (base resolution)
        # For simplicity: shift normalized to base array size
        shift_normalized = self.wind_velocity * time_s / (N * 0.1)  # heuristic scaling
        shift_pixels = shift_normalized * N
        
        # Extract shifted window using np.roll (periodic boundaries)
        # Roll in both x and y
        shifted = np.roll(self._frozen_screen, 
                         shift=(-int(shift_pixels[0]), -int(shift_pixels[1])), 
                         axis=(1, 0))
        
        # Extract central N×N region
        Nlarge = shifted.shape[0]
        start = (Nlarge - N) // 2
        end = start + N
        screen = shifted[start:end, start:end]
        
        return screen

    def process(self, wavefront: Wavefront, context: Context = None) -> Wavefront:
        """Apply atmospheric turbulence to wavefront.
        
        Converts OPD (optical path difference) to phase: φ = 2π * OPD / λ
        This makes the aberration chromatic - shorter wavelengths see larger phase shifts.
        
        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront with wavelength information
        context : Context, optional
            Simulation context (may contain time information)
        
        Returns
        -------
        wavefront : Wavefront
            Wavefront with atmospheric phase applied
        """
        try:
            N = wavefront.field.shape[0]
        except Exception:
            return wavefront

        # Get observation time from context or default to t=0
        if context is not None and hasattr(context, 'time'):
            time = context.time
        else:
            time = 0.0 * u.s
        
        # Extract OPD screen at this time (frozen-flow evolution)
        opd_screen = self._extract_screen_at_time(time, N)
        
        # Convert OPD to phase: φ = 2π * OPD / λ
        # wavefront.wavelength should be in meters
        if hasattr(wavefront, 'wavelength') and wavefront.wavelength is not None:
            if hasattr(wavefront.wavelength, 'to'):
                wavelength_m = wavefront.wavelength.to(u.m).value
            else:
                wavelength_m = float(wavefront.wavelength)
        else:
            # Default wavelength if not specified (550 nm, visible)
            wavelength_m = 550e-9
        
        # Phase in radians
        phase = 2.0 * np.pi * opd_screen / wavelength_m
        
        # Apply phase screen (pure phase modulation)
        wavefront.field = wavefront.field * np.exp(1j * phase).astype(wavefront.field.dtype)
        return wavefront

    def plot_screen_animation(self,
                             collectors: Optional[Union['Collectors', 'Interferometer', List['Collectors']]] = None,
                             times: Optional[np.ndarray] = None,
                             wavelength: u.Quantity = 550e-9*u.m,
                             npix: int = 512,
                             fps: int = 10,
                             duration: Optional[u.Quantity] = None,
                             filename: Optional[str] = None,
                             show_colorbar: bool = True,
                             figsize: Tuple[float, float] = (10, 10)):
        """Create animation of atmospheric phase screen with optional collector overlays.
        
        Displays the temporal evolution of the turbulent phase screen with frozen-flow,
        optionally overlaying collector apertures. The screen extent automatically adjusts
        to show all collectors with 20% margin.
        
        Parameters
        ----------
        collectors : Collectors, Interferometer, list of Collectors, or None
            Collector configuration(s) to overlay. If None, shows phase screen only.
            - Single Collectors: one telescope aperture
            - Interferometer: all baseline-separated apertures
            - List of Collectors: multiple independent telescopes
        times : ndarray, optional
            Observation times in seconds. Auto-generated if None.
        wavelength : astropy.Quantity
            Wavelength for phase calculation (default: 550nm).
        npix : int
            Phase screen resolution (default: 512).
        fps : int
            Animation frame rate (default: 10).
        duration : astropy.Quantity, optional
            Total duration in seconds (default: 5s).
        filename : str, optional
            Save path for animation (requires ffmpeg/pillow).
        show_colorbar : bool
            Show phase colorbar.
        figsize : Tuple[float, float]
            Figure size in inches.
        
        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            Animation object.
        
        Examples
        --------
        >>> # Phase screen only
        >>> atm = Atmosphere(rms=100*u.nm, wind_speed=10*u.m/u.s)
        >>> anim = atm.plot_screen_animation()
        >>> 
        >>> # With interferometer
        >>> vlti = Interferometer(name='VLTI')
        >>> for pos in [(0, 0), (47, 0), (47, 47), (0, 47)]:
        >>>     vlti.add_collector(Pupil.like('VLT'), position=pos, size=8*u.m)
        >>> anim = atm.plot_screen_animation(collectors=vlti, duration=3*u.s)
        """
        from matplotlib.animation import FuncAnimation
        
        # Parse duration
        if duration is None:
            duration = 5.0 * u.s
        duration_s = duration.to(u.s).value if hasattr(duration, 'to') else float(duration)
        
        # Generate time array
        if times is None:
            n_frames = int(fps * duration_s)
            times = np.linspace(0, duration_s, n_frames)
        else:
            times = np.asarray(times)
        
        # Parse wavelength
        wavelength_m = wavelength.to(u.m).value if hasattr(wavelength, 'to') else float(wavelength)
        
        # Extract collector list from input
        collector_list = []
        array_name = "Atmospheric Phase Screen"
        
        if collectors is not None:
            if isinstance(collectors, Interferometer):
                collector_list = collectors.collectors
                array_name = f"{collectors.name}"
            elif isinstance(collectors, list):
                for c_obj in collectors:
                    if hasattr(c_obj, 'collectors'):
                        collector_list.extend(c_obj.collectors)
                array_name = f"{len(collector_list)} collectors"
            elif hasattr(collectors, 'collectors'):
                # Single Collectors object
                collector_list = collectors.collectors
                array_name = "Collectors"
        
        # Determine screen extent based on collectors (with 20% margin)
        if len(collector_list) > 0:
            # Find bounding box of all collectors
            min_x, max_x = 0, 0
            min_y, max_y = 0, 0
            
            for col in collector_list:
                pos = col.get("position", (0, 0))
                size = col.get("size", 1*u.m)
                size_m = size.to(u.m).value if hasattr(size, 'to') else float(size)
                radius = size_m / 2.0
                
                min_x = min(min_x, pos[0] - radius)
                max_x = max(max_x, pos[0] + radius)
                min_y = min(min_y, pos[1] - radius)
                max_y = max(max_y, pos[1] + radius)
            
            # Add 20% margin
            width = max_x - min_x
            height = max_y - min_y
            margin_x = width * 0.2
            margin_y = height * 0.2
            
            extent_x = [min_x - margin_x, max_x + margin_x]
            extent_y = [min_y - margin_y, max_y + margin_y]
            
            # Make square extent (use max dimension)
            max_dim = max(extent_x[1] - extent_x[0], extent_y[1] - extent_y[0])
            center_x = (extent_x[0] + extent_x[1]) / 2.0
            center_y = (extent_y[0] + extent_y[1]) / 2.0
            
            extent = [center_x - max_dim/2, center_x + max_dim/2,
                     center_y - max_dim/2, center_y + max_dim/2]
        else:
            # No collectors: use default extent
            default_extent = 10.0  # meters
            extent = [-default_extent, default_extent, -default_extent, default_extent]
        
        # Create figure
        fig, ax = _plt.subplots(figsize=figsize)
        
        # Mock context for time evolution
        class TimeContext:
            def __init__(self, t):
                self.time = t * u.s
        
        # Generate initial phase screen
        wf_init = Wavefront(wavelength=wavelength, size=npix)
        wf_init.field = np.ones((npix, npix), dtype=np.complex128)
        ctx_init = TimeContext(times[0])
        wf_atm_init = self.process(wf_init, ctx_init)
        phase_init = np.angle(wf_atm_init.field)
        
        # Display initial phase screen
        im = ax.imshow(phase_init, origin='lower', cmap='twilight',
                      extent=extent, vmin=-np.pi, vmax=np.pi,
                      interpolation='bilinear')
        
        # Overlay collector apertures
        for col in collector_list:
            pupil = col.get("pupil", col.get("shape", None))
            if pupil is None:
                continue
            
            pos = col.get("position", (0, 0))
            
            # Render pupil at high resolution
            npix_pupil = 256
            pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
            
            # Create RGBA overlay (white with transparency)
            overlay = np.zeros((npix_pupil, npix_pupil, 4), dtype=float)
            overlay[..., :3] = 1.0  # white
            overlay[..., 3] = pupil_arr * 0.8  # alpha
            
            # Physical extent of this pupil
            diam = pupil.diameter
            extent_pupil = [pos[0] - diam/2, pos[0] + diam/2,
                           pos[1] - diam/2, pos[1] + diam/2]
            
            ax.imshow(overlay, origin='lower', extent=extent_pupil,
                     zorder=10, interpolation='bilinear')
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        
        title = ax.set_title(
            f'{array_name}\\n'
            f't={times[0]:.2f}s, λ={wavelength_m*1e9:.0f}nm, '
            f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s'
        )
        
        if show_colorbar:
            _plt.colorbar(im, ax=ax, label='Phase (radians)', fraction=0.046, pad=0.04)
        
        # Animation update function
        def update(frame_idx):
            t = times[frame_idx]
            
            # Generate phase screen at time t
            wf = Wavefront(wavelength=wavelength, size=npix)
            wf.field = np.ones((npix, npix), dtype=np.complex128)
            ctx = TimeContext(t)
            wf_atm = self.process(wf, ctx)
            phase = np.angle(wf_atm.field)
            
            # Update phase screen
            im.set_data(phase)
            
            # Update title
            title.set_text(
                f'{array_name}\\n'
                f't={t:.2f}s, λ={wavelength_m*1e9:.0f}nm, '
                f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s'
            )
            
            return [im, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(times),
                           interval=1000.0/fps, blit=False, repeat=True)
        
        # Save if requested
        if filename is not None:
            try:
                anim.save(filename, fps=fps, dpi=100)
                print(f"Animation saved to {filename}")
            except Exception as e:
                print(f"Warning: Could not save animation: {e}")
        
        _plt.tight_layout()
        return anim

    def plot_animation(self, 
                      collectors: Union['Collectors', 'Interferometer', List['Collectors']], 
                      times: Optional[np.ndarray] = None,
                      wavelength: u.Quantity = 550e-9*u.m,
                      npix: int = 512,
                      fps: int = 10,
                      duration: Optional[u.Quantity] = None,
                      filename: Optional[str] = None,
                      show_colorbar: bool = True,
                      figsize: Tuple[float, float] = (8, 8)):
        """Create an animation of atmospheric phase screen with collectors overlay.
        
        Shows the temporal evolution of the turbulent phase screen as it drifts
        with the wind (frozen-flow), with the aperture geometry of collectors
        superimposed for reference.
        
        Parameters
        ----------
        collectors : Collectors, Interferometer, or list of Collectors
            Collector configuration(s) to overlay on the phase screen.
            - Single Collectors instance: shows one telescope aperture
            - Interferometer: shows all baseline-separated apertures
            - List of Collectors: shows multiple independent telescopes
        times : ndarray, optional
            Array of observation times in seconds. If None, generates evenly
            spaced times from 0 to duration.
        wavelength : astropy.Quantity
            Wavelength for phase calculation (default: 550nm visible).
        npix : int
            Resolution of the phase screen array (default: 512).
        fps : int
            Frames per second for the animation (default: 10).
        duration : astropy.Quantity, optional
            Total duration of the animation in seconds. If None, uses 5 seconds.
        filename : str, optional
            If provided, saves animation to this file (e.g., 'atm_animation.mp4').
            Requires ffmpeg or pillow for saving.
        show_colorbar : bool
            Whether to show colorbar for phase values.
        figsize : Tuple[float, float]
            Figure size in inches (width, height).
        
        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object. Call plt.show() to display.
        
        Examples
        --------
        >>> # Single telescope
        >>> atm = Atmosphere(rms=100*u.nm, wind_speed=10*u.m/u.s)
        >>> collectors = Collectors()
        >>> collectors.add(pupil=Pupil.like('VLT'), position=(0, 0), size=8*u.m)
        >>> anim = atm.plot_animation(collectors, duration=3*u.s)
        >>> plt.show()
        >>> 
        >>> # Interferometer array
        >>> interferometer = Interferometer(name='VLTI')
        >>> for pos in [(0, 0), (47, 0), (47, 47), (0, 47)]:
        >>>     interferometer.add_collector(Pupil.like('VLT'), position=pos, size=8*u.m)
        >>> anim = atm.plot_animation(interferometer, duration=5*u.s)
        """
        from matplotlib.animation import FuncAnimation
        
        # Parse duration
        if duration is None:
            duration = 5.0 * u.s
        duration_s = duration.to(u.s).value if hasattr(duration, 'to') else float(duration)
        
        # Generate time array if not provided
        if times is None:
            n_frames = int(fps * duration_s)
            times = np.linspace(0, duration_s, n_frames)
        else:
            times = np.asarray(times)
        
        # Parse wavelength
        wavelength_m = wavelength.to(u.m).value if hasattr(wavelength, 'to') else float(wavelength)
        
        # Normalize collectors input to a list
        if isinstance(collectors, Interferometer):
            # Extract aperture configuration from interferometer
            collector_list = collectors.collectors
            array_name = collectors.name
        elif isinstance(collectors, list):
            # List of Collectors objects
            collector_list = []
            for c_obj in collectors:
                if hasattr(c_obj, 'collectors'):
                    collector_list.extend(c_obj.collectors)
            array_name = f"{len(collector_list)} collectors"
        else:
            # Single Collectors object
            if hasattr(collectors, 'collectors'):
                collector_list = collectors.collectors
                array_name = "Collectors"
            else:
                raise TypeError("collectors must be Collectors, Interferometer, or list of Collectors")
        
        # Determine physical extent for the plot
        max_extent = 1.0  # meters, default
        for col in collector_list:
            pos = col.get("position", (0, 0))
            size = col.get("size", 1*u.m)
            size_m = size.to(u.m).value if hasattr(size, 'to') else float(size)
            max_extent = max(max_extent, abs(pos[0]) + size_m/2, abs(pos[1]) + size_m/2)
        
        # Add margin
        max_extent *= 1.2
        
        # Create figure
        fig, ax = _plt.subplots(figsize=figsize)
        
        # Mock context for time evolution
        class TimeContext:
            def __init__(self, t):
                self.time = t * u.s
        
        # Initialize with first frame
        wf_init = Wavefront(wavelength=wavelength, size=npix)
        wf_init.field = np.ones((npix, npix), dtype=np.complex128)
        ctx_init = TimeContext(times[0])
        wf_atm_init = self.process(wf_init, ctx_init)
        phase_init = np.angle(wf_atm_init.field)
        
        # Plot initial phase screen
        im = ax.imshow(phase_init, origin='lower', cmap='twilight', 
                      extent=[-max_extent, max_extent, -max_extent, max_extent],
                      vmin=-np.pi, vmax=np.pi, interpolation='bilinear')
        
        # Overlay collector apertures
        pupil_overlays = []
        for col in collector_list:
            pupil = col.get("pupil", col.get("shape", None))
            if pupil is None:
                continue
            
            pos = col.get("position", (0, 0))
            
            # Render pupil at higher resolution for better visibility
            npix_pupil = 256
            pupil_arr = pupil.get_array(npix=npix_pupil, soft=True)
            
            # Create RGBA overlay (white aperture with transparency)
            overlay = np.zeros((npix_pupil, npix_pupil, 4), dtype=float)
            overlay[..., :3] = 1.0  # white
            overlay[..., 3] = pupil_arr * 0.7  # alpha channel (increased for better visibility)
            
            # Physical extent of pupil
            diam = pupil.diameter
            extent_pupil = [pos[0] - diam/2, pos[0] + diam/2, 
                           pos[1] - diam/2, pos[1] + diam/2]
            
            overlay_im = ax.imshow(overlay, origin='lower', extent=extent_pupil, 
                                  zorder=10, interpolation='bilinear')
            pupil_overlays.append(overlay_im)
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        
        title = ax.set_title(f'Atmospheric Phase Screen - {array_name}\\n' + 
                            f't={times[0]:.2f}s, λ={wavelength_m*1e9:.0f}nm, ' + 
                            f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s')
        
        if show_colorbar:
            cbar = _plt.colorbar(im, ax=ax, label='Phase (radians)', fraction=0.046, pad=0.04)
        
        # Animation update function
        def update(frame_idx):
            t = times[frame_idx]
            
            # Generate phase screen at time t
            wf = Wavefront(wavelength=wavelength, size=npix)
            wf.field = np.ones((npix, npix), dtype=np.complex128)
            ctx = TimeContext(t)
            wf_atm = self.process(wf, ctx)
            phase = np.angle(wf_atm.field)
            
            # Update image data
            im.set_data(phase)
            
            # Update title
            title.set_text(f'Atmospheric Phase Screen - {array_name}\\n' + 
                          f't={t:.2f}s, λ={wavelength_m*1e9:.0f}nm, ' + 
                          f'OPD RMS={self.rms*1e9:.0f}nm, wind={np.linalg.norm(self.wind_velocity):.1f}m/s')
            
            return [im, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(times), 
                           interval=1000.0/fps, blit=False, repeat=True)
        
        # Save if filename provided
        if filename is not None:
            try:
                anim.save(filename, fps=fps, dpi=100)
                print(f"Animation saved to {filename}")
            except Exception as e:
                print(f"Warning: Could not save animation: {e}")
        
        _plt.tight_layout()
        return anim



class AdaptiveOptics(Layer):
    """Adaptive optics layer applying Zernike-based correction.

    - `coeffs`: mapping from (n,m) -> coefficient in radians. `n` >= 0, `m` integer with |m|<=n and (n-|m|) even.
      Example: {(1,1): 0.1} for Zernike n=1,m=1.
    - `normalize`: whether to evaluate Zernikes on unit pupil mapped to array size.
    """
    def __init__(self, coeffs: Optional[dict] = None, normalize: bool = True):
        super().__init__()
        self.coeffs = coeffs or {}
        self.normalize = normalize

    @staticmethod
    def noll_to_nm(j: int) -> Tuple[int, int]:
        """Convert Noll index (1-based) to Zernike (n,m).

        This uses the standard Noll ordering. Returns (n,m).
        """
        if j < 1:
            raise ValueError("Noll index must be >= 1")
        # Noll indexing: j=1 -> (0,0); j=2 -> (1,-1); j=3 -> (1,1); j=4 -> (2,-2) ...
        # We'll compute by enumerating until reach index j.
        count = 0
        n = 0
        while True:
            for m in range(-n, n + 1, 2):
                count += 1
                if count == j:
                    return (n, m)
            n += 1

    def _radial_polynomial(self, n: int, m: int, r: np.ndarray) -> np.ndarray:
        m = abs(m)
        if (n - m) % 2 != 0:
            return np.zeros_like(r)
        R = np.zeros_like(r)
        kmax = (n - m) // 2
        for k in range(kmax + 1):
            num = (-1) ** k * math.factorial(n - k)
            den = math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
            R += num / den * r ** (n - 2 * k)
        return R

    def _zernike_nm(self, n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # m may be negative: negative -> sin component
        if m == 0:
            R = self._radial_polynomial(n, 0, rho)
            return R
        elif m > 0:
            R = self._radial_polynomial(n, m, rho)
            return R * np.cos(m * theta)
        else:
            R = self._radial_polynomial(n, -m, rho)
            return R * np.sin((-m) * theta)

    def process(self, wavefront: Wavefront, context: Context) -> Wavefront:
        try:
            N = wavefront.field.shape[0]
        except Exception:
            return wavefront

        # coordinates normalized to unit disk
        ys = np.linspace(-1.0, 1.0, N)
        xs = ys.copy()
        xg, yg = np.meshgrid(xs, ys)
        rho = np.hypot(xg, yg)
        theta = np.arctan2(yg, xg)
        mask = rho <= 1.0

        # build AO correction phase
        phase = np.zeros((N, N), dtype=float)
        # allow coeff keys to be either (n,m) tuples or Noll integer indices
        items = []
        for k, coeff in self.coeffs.items():
            if isinstance(k, int):
                nm = self.noll_to_nm(k)
            else:
                nm = tuple(k)
            items.append((nm, coeff))

        for (n, m), coeff in items:
            if hasattr(coeff, 'to'):
                c = float(coeff.to(u.rad).value)
            else:
                c = float(coeff)
            Z = self._zernike_nm(n, m, rho, theta)
            phase += c * Z

        # apply only inside pupil (unit disk)
        phase = phase * mask
        # AO subtracts estimated phase (apply negative phase)
        wavefront.field = wavefront.field * np.exp(-1j * phase).astype(wavefront.field.dtype)
        return wavefront

def test_collectors():
    c = Collectors(latitude=0*u.deg, longitude=0*u.deg, altitude=0*u.m)
    p = Pupil(1*u.m)
    c.add(size=8*u.m, pupil=p, position=(0, 0))
    assert len(c.collectors) == 1

    # Test defaults
    default_c = Collectors()
    assert default_c.latitude == 0*u.deg
    assert default_c.longitude == 0*u.deg
    assert default_c.altitude == 0*u.m

if __name__ == "__main__":
    test_collectors()
    print("Optics tests passed.")
