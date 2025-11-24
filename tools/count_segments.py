import sys
import numpy as np
sys.path.insert(0, r'e:/HELIOS/src')
from astropy import units as u

D = 39.3
primR = D/2.0

def count_segments(seg_flat, rings):
    a = seg_flat / np.sqrt(3.0)
    centers = []
    N = rings
    for q in range(-N, N + 1):
        r1 = max(-N, -q - N)
        r2 = min(N, -q + N)
        for r in range(r1, r2 + 1):
            x = a * 3.0 / 2.0 * q
            y = a * np.sqrt(3.0) * (r + q / 2.0)
            centers.append((x, y))
    cnt = sum(1 for (cx, cy) in centers if np.hypot(cx, cy) <= primR + 1e-12)
    return cnt

if __name__ == '__main__':
    rings = 16
    targets = []
    for sf in np.linspace(1.0, 1.6, 61):
        cnt = count_segments(sf, rings)
        targets.append((sf, cnt))
    targets.sort(key=lambda x: abs(x[1] - 798))
    print('Top candidates (seg_flat -> count) for rings=16:')
    for sf, cnt in targets[:10]:
        print(f'seg_flat={sf:.3f} -> {cnt}')
    # also try varying rings small range
    print('\nSearch over rings and seg_flat grid:')
    best = []
    for r in range(12, 22):
        for sf in np.linspace(1.0, 1.6, 61):
            cnt = count_segments(sf, r)
            best.append((r, sf, cnt, abs(cnt - 798)))
    best.sort(key=lambda x: x[3])
    for r, sf, cnt, diff in best[:10]:
        print(f'rings={r}, seg_flat={sf:.3f} -> {cnt} (diff={diff})')
