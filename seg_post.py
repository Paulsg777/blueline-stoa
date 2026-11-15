"""Post-processing of segmented rooms, using the labelled rooms from the language model as evidence.

split_merged: a segment that two or more labelled rooms fall into is two rooms joined through an open doorway.
  Erode it until it falls apart at the narrow neck (the doorway), then give every pixel back to the nearest core.
  If it has no neck (open plan), split it between the labelled rooms by nearest distance instead.
total_indoor: total indoor area from the segments themselves (every segment not under an outdoor room),
  so rooms the language model missed still count.
"""
import numpy as np, cv2
from shapely.geometry import Polygon

OUT = ("balkon", "balcon", "loggia", "terrass", "terrazz", "veranda", "sitzplatz", "parveke", "terassi")


def P(poly):
    try:
        p = Polygon(poly).buffer(0)
        return p if p.area > 0 else None
    except Exception:
        return None


def _mask(poly, box):
    x0, y0, x1, y1 = box
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly) - [x0, y0]).astype(np.int32)], 1)
    return m


def _pieces(lbl, box, keep):
    out = []
    for k in keep:
        one = (lbl == k).astype(np.uint8)
        cs, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        c = max(cs, key=cv2.contourArea)
        poly = (cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) + [box[0], box[1]]).tolist()
        if len(poly) >= 3: out.append({"polygon": poly, "area_px": int(one.sum())})
    return out


def _nearest(m, seeds):
    """Give every pixel of m to the nearest seed mask; returns a label image (0 = outside)."""
    d = np.stack([cv2.distanceTransform((1 - s).astype(np.uint8), cv2.DIST_L2, 3) for s in seeds])
    lbl = d.argmin(0) + 1
    lbl[m == 0] = 0
    return lbl


def split_merged(rooms, segs, ppm):
    ppm = ppm or 30.0
    rp = [P(r["polygon"]) if len(r.get("polygon") or []) >= 3 else None for r in rooms]
    out, nsplit = [], 0
    for s in segs:
        sp = P(s["polygon"])
        hits = [i for i, q in enumerate(rp) if q is not None and sp is not None and q.intersects(sp)
                and q.intersection(sp).area > 0.3 * q.area]
        if sp is None or len(hits) < 2:
            out.append(s); continue
        x0, y0, x1, y1 = [int(v) for v in sp.bounds]; box = (x0 - 2, y0 - 2, x1 + 3, y1 + 3)
        m = _mask(s["polygon"], box)
        r = max(2, int(0.35 * ppm))                                   # doorways are ~0.8-1 m wide
        core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        n, cc, st, _ = cv2.connectedComponentsWithStats(core, 4)
        cores = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > (0.5 * ppm) ** 2]
        if len(cores) >= 2:
            lbl = _nearest(m, [(cc == k).astype(np.uint8) for k in cores])
        else:                                                         # no neck: split between the labelled rooms
            lbl = _nearest(m, [_mask(rooms[i]["polygon"], box) & m for i in hits])
        pcs = _pieces(lbl, box, range(1, lbl.max() + 1))
        if len(pcs) >= 2: out += pcs; nsplit += 1
        else: out.append(s)
    return out, nsplit


def total_indoor(rooms, segs, ppm):
    """Sum of segment areas, leaving out segments mostly under an outdoor labelled room."""
    if not ppm: return None
    outdoor = [P(r["polygon"]) for r in rooms if any(w in (r.get("label") or "").lower() for w in OUT)
               and len(r.get("polygon") or []) >= 3]
    tot = 0.0
    for s in segs:
        sp = P(s["polygon"])
        if sp is None: continue
        if any(o is not None and o.intersects(sp) and o.intersection(sp).area > 0.5 * sp.area for o in outdoor): continue
        tot += s.get("area_px", sp.area)
    return round(tot / ppm ** 2, 2)
