import numpy as np

D = 39.3
primR = D/2.0

# function to count segments for given seg_flat, rings, central_rings, n_sides=12
def count_segments(seg_flat, rings, central_rings, n_sides=12):
    a = seg_flat / np.sqrt(3.0)
    # generate poly vertices for regular n_sides polygon
    thetas = np.linspace(0.0, 2.0*np.pi, n_sides, endpoint=False)
    poly_verts = [(primR * np.cos(t), primR * np.sin(t)) for t in thetas]
    from matplotlib.path import Path
    poly_path = Path(np.asarray(poly_verts, dtype=float))

    centers = []
    N = rings
    for q in range(-N, N+1):
        r1 = max(-N, -q - N)
        r2 = min(N, -q + N)
        for r in range(r1, r2+1):
            cx = a * 3.0 / 2.0 * q
            cy = a * np.sqrt(3.0) * (r + q / 2.0)
            if poly_path.contains_point((cx, cy)):
                centers.append((cx, cy))
    total = len(centers)
    # count central occluder centers
    central = []
    Ncent = central_rings
    for q in range(-Ncent, Ncent+1):
        r1 = max(-Ncent, -q - Ncent)
        r2 = min(Ncent, -q + Ncent)
        for r in range(r1, r2+1):
            cx = a * 3.0 / 2.0 * q
            cy = a * np.sqrt(3.0) * (r + q / 2.0)
            central.append((cx, cy))
    # number of transmissive segments = total - count of centers that are in central set and also in centers
    centra_set = set((round(x,6), round(y,6)) for x,y in central)
    centers_set = set((round(x,6), round(y,6)) for x,y in centers)
    overlap = centers_set & centra_set
    transmissive = total - len(overlap)
    return transmissive, total, len(overlap)

if __name__ == '__main__':
    best = []
    for rings in range(14, 19):
        for sf in np.linspace(1.20, 1.40, 81):
            for central_rings in [3,4,5]:
                t, tot, over = count_segments(sf, rings, central_rings)
                if t == 798:
                    print(f'Found exact: seg_flat={sf:.3f}, rings={rings}, central_rings={central_rings} -> transmissive={t} (total_before={tot}, occluded={over})')
                    raise SystemExit(0)
                best.append((abs(t-798), sf, rings, central_rings, t, tot, over))
    best.sort(key=lambda x: x[0])
    print('Top candidates:')
    for diff, sf, rings, central_rings, t, tot, over in best[:10]:
        print(f'diff={diff} seg_flat={sf:.3f} rings={rings} central_rings={central_rings} -> {t} (total={tot} occl={over})')
