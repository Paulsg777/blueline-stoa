"""Find the door swing arc near a point: the one drawing feature a door has and a window does not.

A door is drawn as a quarter circle around its hinge, radius = the door's width (about 0.6-1.2 m). For every radius
and each of the 4 quadrants, a thin quarter-ring kernel is slid over the drawing's ink near the point; the score is
the fraction of the ring that is inked. A straight line (wall, window, hatch) covers only a sliver of a quarter ring,
a door's arc covers nearly all of it.
"""
import numpy as np, cv2

WORK_PPM = 59.0
_K = {}


def _kernels(r):
    """Per quadrant: +1/len on the quarter ring, -1/len on rings 5 px inside and outside it. An isolated arc scores ~1;
    a filled or hatched area (ink everywhere) scores ~0; a straight line crossing the ring scores ~0."""
    if r not in _K:
        ks = []
        n = 2 * r + 15
        for q in range(4):
            a0 = q * 90
            on, off = np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
            cv2.ellipse(on, (n // 2, n // 2), (r, r), 0, a0 + 8, a0 + 82, 1.0, 1)      # trim the ends: they touch wall/leaf
            for rr in (r - 5, r + 5):
                cv2.ellipse(off, (n // 2, n // 2), (rr, rr), 0, a0 + 8, a0 + 82, 1.0, 1)
            ks.append(on / on.sum() - off / off.sum())
        _K[r] = ks
    return _K[r]


def best_arc(img, pt, ppm, search_m=1.2, rmin_m=0.55, rmax_m=1.2, gray=None, exclude=()):
    """Strongest door arc whose symbol lies near pt (image px). Returns (score, box [x0,y0,x1,y1] in image px) or None."""
    s = WORK_PPM / ppm if ppm else 1.0
    g = gray if gray is not None else (img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    R = int((search_m + rmax_m) * ppm) if ppm else 150
    x, y = int(pt[0]), int(pt[1])
    X0, Y0 = max(0, x - R), max(0, y - R)
    c = g[Y0:y + R, X0:x + R]
    if c.size == 0: return None
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if abs(s - 1) > 0.02 else c
    ink = (c < 170).astype(np.uint8)
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))        # walls and fills: not arcs
    thin = (ink & (1 - cv2.dilate(thick, np.ones((3, 3), np.uint8)))).astype(np.float32)
    thin = cv2.dilate(thin, np.ones((2, 2), np.uint8))
    for b in exclude:                                       # symbols already counted: their arcs are not candidates
        x0, y0, x1, y1 = [int(round((v - o) * s)) for v, o in zip(b, (X0, Y0, X0, Y0))]
        thin[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    px, py = (x - X0) * s, (y - Y0) * s
    best = None
    for r in range(int(rmin_m * WORK_PPM), int(rmax_m * WORK_PPM) + 1, 3):
        for q, k in enumerate(_kernels(r)):
            resp = cv2.filter2D(thin, -1, k, borderType=cv2.BORDER_CONSTANT)
            # hinge candidates whose quarter-disc centre is within search_m of the point
            dx, dy = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)][q]          # quadrant 0 = +x,+y (image coords)
            yy, xx = np.mgrid[0:resp.shape[0], 0:resp.shape[1]]
            near = np.hypot(xx + dx * r - px, yy + dy * r - py) <= search_m * WORK_PPM
            resp = np.where(near, resp, 0)
            iy, ix = np.unravel_index(int(resp.argmax()), resp.shape)
            sc = float(resp[iy, ix])
            if best is None or sc > best[0]:
                bx = [ix, iy, ix + 2 * dx * r, iy + 2 * dy * r]
                box = [min(bx[0], bx[2]), min(bx[1], bx[3]), max(bx[0], bx[2]), max(bx[1], bx[3])]
                best = (sc, [X0 + box[0] / s, Y0 + box[1] / s, X0 + box[2] / s, Y0 + box[3] / s], (X0 + ix / s, Y0 + iy / s))
    return best


def leaves(img, box, ppm, gray=None, thr=0.6):
    """How many door leaves (swing arcs) a detected door opening has: 1, or 2 for a double / French door.
    A second arc counts only if it is as clear as a door arc and swings from a different hinge."""
    c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    a = best_arc(img, c, ppm, search_m=0.8, gray=gray)
    if not a or a[0] < thr: return 1, [a] if a else []
    b = best_arc(img, c, ppm, search_m=0.8, gray=gray, exclude=[a[1]])
    if b and b[0] >= thr:
        (ha, hb), p = (a[2], b[2]), (ppm or 59)
        ra, rb = a[1][2] - a[1][0], b[1][2] - b[1][0]
        dx, dy = abs(ha[0] - hb[0]), abs(ha[1] - hb[1])
        collinear = min(dx, dy) <= 0.15 * p                 # both hinges on the same wall line
        meet = abs(max(dx, dy) - (ra + rb)) <= 0.25 * (ra + rb)   # the two leaves meet in the middle of one opening
        if collinear and meet:
            return 2, [a, b]
    return 1, [a]
