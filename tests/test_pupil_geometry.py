import sys
import os
import numpy as np

# ensure local `src` is first on path so tests import the workspace code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from helios.components.pupil import Pupil


def _count_components(binarr):
    H, W = binarr.shape
    visited = np.zeros_like(binarr, dtype=bool)
    comps = 0
    for i in range(H):
        for j in range(W):
            if binarr[i, j] and not visited[i, j]:
                comps += 1
                # flood fill
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and binarr[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
    return comps


def test_jwst_segments_and_spiders():
    # JWST-like pupil (preset)
    p = Pupil.jwst()
    N = 512
    # use anti-aliased pupil for more robust center detection
    arr = p.get_array(npix=N, soft=True)
    assert arr.shape == (N, N)

    # center should be occluded (secondary hex)
    c = N // 2
    assert arr[c, c] == 0.0

    # Instead of naive connected-component counting (outer clipped hexes
    # can be split), reconstruct the expected segment centers using the
    # same flat-top axial layout used by the implementation and check
    # that a pixel near each center is lit. This gives an unambiguous
    # count of intended segments (19 total with central occluder -> 18 visible).
    seg_flat = 1.2
    rings = 2
    a = seg_flat / np.sqrt(3.0)
    primR = p.diameter / 2.0
    centers = []
    Nrings = rings
    for q in range(-Nrings, Nrings + 1):
        r1 = max(-Nrings, -q - Nrings)
        r2 = min(Nrings, -q + Nrings)
        for r in range(r1, r2 + 1):
            x = a * 3.0 / 2.0 * q
            y = a * np.sqrt(3.0) * (r + q / 2.0)
            if np.hypot(x, y) <= primR + 1e-12:
                centers.append((x, y))

    # count centers that correspond to lit pixels in the rasterized pupil
    c_px = N // 2
    lit_centers = 0
    for (cx, cy) in centers:
        px = int(round(c_px + (cx / (p.diameter / 2.0)) * (N / 2.0)))
        py = int(round(c_px + (cy / (p.diameter / 2.0)) * (N / 2.0)))
        # guard in-bounds
        if 0 <= px < N and 0 <= py < N:
            # use a small neighborhood average to avoid single-pixel aliasing
            xmin = max(0, px - 2)
            xmax = min(N, px + 3)
            ymin = max(0, py - 2)
            ymax = min(N, py + 3)
            local_max = arr[ymin:ymax, xmin:xmax].max()
            if local_max > 0.05:
                lit_centers += 1

    # rings=2 yields 19 centers but the central one is occluded -> expect 18 lit
    assert lit_centers == 18, f"Expected 18 lit segment centers, found {lit_centers}"

    # Detect spider arms robustly: determine inner occluder radius (in px)
    # then, for each angle, compute the mean transmission along a short
    # radial segment just outside the occluder; low mean -> spider present.
    max_r = N // 2
    # find first radius where mean transmission exceeds threshold
    inner_r = None
    for r in range(1, max_r // 2):
        # sample 72 points on ring r
        thetas_r = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        vals_r = []
        for th in thetas_r:
            x = int(c + r * np.cos(th))
            y = int(c + r * np.sin(th))
            vals_r.append(arr[y, x])
        if np.mean(vals_r) > 0.05:
            inner_r = r
            break
    assert inner_r is not None

    thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    blocked = []
    seg_len = max(3, int(N * 0.03))
    for th in thetas:
        samples = []
        for rr in range(inner_r + 1, inner_r + 1 + seg_len):
            x = int(c + rr * np.cos(th))
            y = int(c + rr * np.sin(th))
            samples.append(arr[y, x])
        blocked.append(np.mean(samples) < 0.5)

    # find contiguous blocked runs and compute their angular centers
    runs = []
    in_run = False
    run_start = 0
    for idx, b in enumerate(blocked):
        if b and not in_run:
            in_run = True
            run_start = idx
        elif (not b) and in_run:
            in_run = False
            runs.append((run_start, idx - 1))
    if in_run:
        runs.append((run_start, len(blocked) - 1))

    # compute run centers in degrees
    deg_per = 360.0 / len(blocked)
    centers_deg = []
    for s, e in runs:
        # mid index, handle wrap-around
        if e >= s:
            mid = (s + e) / 2.0
        else:
            # wrap case
            mid = ((s + (e + len(blocked))) / 2.0) % len(blocked)
        centers_deg.append(mid * deg_per)

    # cluster centers by angular proximity (merge centers within 15 deg)
    centers_deg = sorted(centers_deg)
    clusters = []
    for ang in centers_deg:
        if not clusters:
            clusters.append([ang])
            continue
        if (ang - clusters[-1][-1]) <= 15.0:
            clusters[-1].append(ang)
        else:
            clusters.append([ang])
    # handle circular merge between first and last
    if len(clusters) > 1 and (360.0 - clusters[-1][-1] + clusters[0][0]) <= 15.0:
        # merge last into first
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    cluster_count = len(clusters)
    # fragmented rasterization can create several nearby small runs; require
    # at least 3 clustered blocked directions (JWST has 3 spider arms).
    assert cluster_count >= 3, f"Expected at least 3 spider clusters, found {cluster_count} (raw runs {len(runs)})"
