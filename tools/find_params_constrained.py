import numpy as np
from matplotlib.path import Path

# fixed geometry per user's current configuration
D = 39.3
primR = D / 2.0
rings = 30 # Ensure generation covers the dodecagon
central_rings = 5 # User requested adding a ring to secondary (was 4)
n_sides = 12

# polygon circumradius so that polygon apothem == primR
circum_r = primR / np.cos(np.pi / float(n_sides))
thetas = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
poly_verts = [(circum_r * np.cos(t), circum_r * np.sin(t)) for t in thetas]
poly = Path(np.asarray(poly_verts, dtype=float))

candidates = []

# coarse sweep (wider range)
for seg_flat in np.arange(1.10, 1.50 + 1e-12, 0.002):
    for gap in [0.0, 0.004, 0.01]:
        a = seg_flat / np.sqrt(3.0)
        centers = []
        N = rings
        for q in range(-N, N + 1):
            r1 = max(-N, -q - N)
            r2 = min(N, -q + N)
            for r in range(r1, r2 + 1):
                cx = a * 3.0 / 2.0 * q
                cy = a * np.sqrt(3.0) * (r + q / 2.0)
                if poly.contains_point((cx, cy)):
                    centers.append((cx, cy))
        total = len(centers)
        # central occluder
        a_cent = seg_flat / np.sqrt(3.0)
        centers_c = []
        Ncent = central_rings
        for q in range(-Ncent, Ncent + 1):
            r1 = max(-Ncent, -q - Ncent)
            r2 = min(Ncent, -q + Ncent)
            for r in range(r1, r2 + 1):
                cx = a_cent * 3.0 / 2.0 * q
                cy = a_cent * np.sqrt(3.0) * (r + q / 2.0)
                centers_c.append((cx, cy))
        # count overlap
        setc = set((round(x,6), round(y,6)) for x,y in centers)
        setcen = set((round(x,6), round(y,6)) for x,y in centers_c)
        overlap = len(setc & setcen)
        transmissive = total - overlap
        if transmissive == 798:
            candidates.append((seg_flat, gap, total, overlap, transmissive))

print('Found candidates (coarse):', candidates)

# if none found, try finer sweep around previously found region (if any coarse near 798)
if not candidates:
    # compute closest distances
    best = []
    for seg_flat in np.arange(1.05, 1.35 + 1e-12, 0.005):
        a = seg_flat / np.sqrt(3.0)
        centers = []
        N = rings
        for q in range(-N, N + 1):
            r1 = max(-N, -q - N)
            r2 = min(N, -q + N)
            for r in range(r1, r2 + 1):
                cx = a * 3.0 / 2.0 * q
                cy = a * np.sqrt(3.0) * (r + q / 2.0)
                if poly.contains_point((cx, cy)):
                    centers.append((cx, cy))
        total = len(centers)
        a_cent = seg_flat / np.sqrt(3.0)
        centers_c = []
        Ncent = central_rings
        for q in range(-Ncent, Ncent + 1):
            r1 = max(-Ncent, -q - Ncent)
            r2 = min(Ncent, -q + Ncent)
            for r in range(r1, r2 + 1):
                cx = a_cent * 3.0 / 2.0 * q
                cy = a_cent * np.sqrt(3.0) * (r + q / 2.0)
                centers_c.append((cx, cy))
        setc=set((round(x,6),round(y,6)) for x,y in centers)
        setcen=set((round(x,6),round(y,6)) for x,y in centers_c)
        overlap=len(setc & setcen)
        transmissive = total - overlap
        best.append((abs(transmissive-798), seg_flat, transmissive, total, overlap))
    best_sorted = sorted(best, key=lambda x: x[0])[:10]
    print('Closest coarse results (diff, seg_flat, transmissive, total, overlap):')
    for row in best_sorted:
        print(row)
    # refine around the best seg_flat
    if best_sorted:
        center_seg = best_sorted[0][1]
        lo = max(0.9, center_seg - 0.02)
        hi = center_seg + 0.02
        candidates_fine = []
        for seg_flat in np.arange(lo, hi + 1e-12, 0.001):
            for gap in [0.0, 0.001, 0.005, 0.01]:
                a = seg_flat / np.sqrt(3.0)
                centers = []
                N = rings
                for q in range(-N, N + 1):
                    r1 = max(-N, -q - N)
                    r2 = min(N, -q + N)
                    for r in range(r1, r2 + 1):
                        cx = a * 3.0 / 2.0 * q
                        cy = a * np.sqrt(3.0) * (r + q / 2.0)
                        if poly.contains_point((cx, cy)):
                            centers.append((cx, cy))
                total = len(centers)
                a_cent = seg_flat / np.sqrt(3.0)
                centers_c = []
                Ncent = central_rings
                for q in range(-Ncent, Ncent + 1):
                    r1 = max(-Ncent, -q - Ncent)
                    r2 = min(Ncent, -q + Ncent)
                    for r in range(r1, r2 + 1):
                        cx = a_cent * 3.0 / 2.0 * q
                        cy = a_cent * np.sqrt(3.0) * (r + q / 2.0)
                        centers_c.append((cx, cy))
                setc=set((round(x,6),round(y,6)) for x,y in centers)
                setcen=set((round(x,6),round(y,6)) for x,y in centers_c)
                overlap=len(setc & setcen)
                transmissive = total - overlap
                if transmissive == 798:
                    candidates_fine.append((seg_flat, gap, total, overlap, transmissive))
        print('Found candidates (fine):', candidates_fine)
