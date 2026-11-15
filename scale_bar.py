"""Measure a drawing's scale bar in pixels (general CV, no dataset priors).

find_bars(img, box=None) -> candidate bars [(x0, x1, y, length_px)], best first.
A scale bar is a thin, long, mostly filled dark strip (often alternating black/white segments inside one outline).
If a rough box is given (e.g. from a language model, 0-1 normalised), only that area (+ margin) is searched.
px_per_m = length_px / bar_length_m, where bar_length_m is the last number printed on the bar (read by the language model).
"""
import math
import numpy as np, cv2
MANT = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)   # scale bars end at round numbers: 1, 2, 5, 10, 8 ft, 2000 mm ...


def find_bars(gray, box=None, relaxed=False):
    H, W = gray.shape
    x0 = y0 = 0; g = gray
    if box:
        bx0, by0, bx1, by1 = box; m = 0.05
        x0, y0 = int(max(0, bx0 - m) * W), int(max(0, by0 - m) * H)
        g = gray[y0:int(min(1, by1 + m) * H), x0:int(min(1, bx1 + m) * W)]
    dark = (g < 128).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    cands = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        # size limits relative to the sheet; relaxed (only used inside the model's box) for big sheets with a small bar
        if relaxed:
            if w < 0.01 * W or h < 3 or h > 0.02 * H or w < 8 * h: continue
        elif w < 0.04 * W or h < max(3, 0.0025 * max(W, H)) or h > 0.02 * H or w < 8 * h: continue
        sub = cc[y:y + h, x:x + w] == k
        rows = sub.mean(1)
        if rows.max() < 0.95: continue                       # outline rows must run the full length
        cols = sub.mean(0)
        e = max(2, min(6, h // 4))
        if cols[:e].max() < 0.5 or cols[-e:].max() < 0.5: continue   # closed ends (a mostly dark column near each end)
        if sub[:, w // 4: 3 * w // 4].mean() < 0.3: continue        # a filled strip, not two parallel lines
        cands.append((x + x0, x + w + x0, y + h // 2 + y0, int(w) - 1, a / (w * h)))   # -1: centre of the end strokes
    cands.sort(key=lambda c: (-c[3]))
    if relaxed:                                                 # alternating black/white inside an outline beats a solid rule line
        cands.sort(key=lambda c: (c[4] > 0.95, -c[3]))                                                 # bars drawn as separate black blocks (black/white/black...)
        cands = segmented_bars(st[1:], x0, y0, W, H) + cands
    return cands


def segmented_bars(st, ox, oy, W, H):
    """Scale bars made of separate equal black blocks with equal white gaps between them (no outline joining them):
    a chain of >= 3 blocks of the same height on one line, block widths and gaps alike. Length = first to last edge."""
    blocks = [tuple(int(v) for v in b[:4]) for b in st if 3 <= b[3] <= 0.02 * H and b[2] >= 1.5 * b[3] and b[4] >= 0.85 * b[2] * b[3]]
    blocks.sort(key=lambda b: b[0]); out = []
    used = set()
    for i, b in enumerate(blocks):
        if i in used: continue
        chain = [b]
        for j in range(i + 1, len(blocks)):
            c = blocks[j]; p = chain[-1]
            if abs(c[1] - p[1]) > 2 or abs(c[3] - p[3]) > 2: continue
            gap = c[0] - (p[0] + p[2])
            if gap < 0: continue
            if gap > 3 * p[2]: break                    # sorted by x: nothing further along can continue this chain
            if 0.6 * p[2] <= gap <= 1.4 * p[2] and 0.7 <= c[2] / p[2] <= 1.4:
                chain.append(c); used.add(j)
        if len(chain) >= 3:
            x_0, x_1 = chain[0][0], chain[-1][0] + chain[-1][2]
            out.append((x_0 + ox, x_1 + ox, chain[0][1] + chain[0][3] // 2 + oy, x_1 - x_0 - 1, 1.0))
    out.sort(key=lambda c: -c[3])
    return out


NICE = (0.5, 1, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 64, 75, 80, 100, 120, 150, 160, 200)   # metres or feet


def refine_scale(gray, rough_ppm, box=None, tol=0.12):
    """Precise px/m from the scale bar, given a rough estimate (e.g. a language model's reading).
    The bar's length in metres is the 'nice' number closest to bar_px / rough_ppm; accepted only if within tol."""
    if not rough_ppm: return None
    for x0, x1, y, L, fill in find_bars(gray, box):
        m = L / rough_ppm
        k = 10 ** math.floor(math.log10(m))                       # any unit: 2, 8, 80, 2000 ... (mantissa x power of ten)
        nice = min((n * k * f for n in MANT for f in (1, 10)), key=lambda v: abs(m / v - 1))
        if abs(m / nice - 1) < tol:
            return L / nice
    return None


def segmented_in(gray, box):
    H, W = gray.shape; m = 0.05
    bx0, by0, bx1, by1 = box
    x0, y0 = int(max(0, bx0 - m) * W), int(max(0, by0 - m) * H)
    g = gray[y0:int(min(1, by1 + m) * H), x0:int(min(1, bx1 + m) * W)]
    n, cc, st, _ = cv2.connectedComponentsWithStats((g < 128).astype(np.uint8), 8)
    return segmented_bars(st[1:], x0, y0, W, H)
