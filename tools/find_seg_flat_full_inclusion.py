import numpy as np
from matplotlib.path import Path

D = 39.3
primR = D/2.0
n_sides = 12
circum_r = primR/np.cos(np.pi/n_sides)
poly_path = Path(np.asarray([(circum_r*np.cos(t), circum_r*np.sin(t)) for t in np.linspace(0,2*np.pi,n_sides,endpoint=False)], dtype=float))

rings = 30
central_rings = 5

def counts(seg_flat, gap=0.004):
    a = seg_flat/np.sqrt(3.0)
    a_draw = (seg_flat-gap)/np.sqrt(3.0)
    # build set of occluded centers
    occ_centers = set()
    a_c = seg_flat/np.sqrt(3.0)
    for q in range(-central_rings, central_rings+1):
        r1 = max(-central_rings, -q-central_rings)
        r2 = min(central_rings, -q+central_rings)
        for r in range(r1, r2+1):
            cx = a_c*1.5*q
            cy = a_c*np.sqrt(3.0)*(r + q/2.0)
            occ_centers.add((round(cx,6), round(cy,6)))
    trans = 0
    occ = 0
    for q in range(-rings, rings+1):
        r1 = max(-rings, -q-rings)
        r2 = min(rings, -q+rings)
        for r in range(r1, r2+1):
            cx = a*1.5*q
            cy = a*np.sqrt(3.0)*(r + q/2.0)
            thet = np.linspace(0,2*np.pi,7)
            verts = np.column_stack((cx + a_draw*np.cos(thet), cy + a_draw*np.sin(thet)))
            if all(poly_path.contains_point(v) for v in verts[:-1]):
                if (round(cx,6), round(cy,6)) in occ_centers:
                    occ += 1
                else:
                    trans += 1
    return trans, occ, trans+occ

best = []
found = []
for gap in [0.0, 0.002, 0.004, 0.006, 0.008, 0.01]:
    for seg_flat in np.arange(1.18, 1.26+1e-12, 0.001):
        trans, occ, total = counts(seg_flat, gap=gap)
        if trans == 798:
            found.append((seg_flat, gap, trans, occ, total))
            print('FOUND', seg_flat, gap, trans, occ, total)
            # do not break; collect all
        best.append((abs(trans-798), seg_flat, gap, trans, occ, total))

if not found:
    print('No exact match, closest 10:')
    for row in sorted(best)[:10]:
        print(row)
else:
    print('All matches:')
    for row in found:
        print(row)
