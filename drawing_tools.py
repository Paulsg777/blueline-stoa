"""Deterministic tools for any drawing (floor plans, piping, structural). Each returns numbers plus the geometry it used,
so every answer can be drawn back onto the sheet.

    read_text(img)                      -> [{text, box, conf}]            local OCR (RapidOCR), tiled, de-duplicated
    count_tags(texts, "CV")             -> {distinct, instances, hits}    e.g. CV-1..CV-5 on a piping plan
    count_symbol(img, box)              -> {count, hits}                  every copy of the symbol inside `box`
    scale_from_bar(img, rough)          -> px per unit                     (scale_bar.refine_scale)
    region_area(img, seed, ppu)         -> {area, polygon}                 enclosed area around a seed point
    colour_area(img, rgb, ppu, roi)     -> {area, mask}                    area of a legend colour
    polyline_length(points, ppu)        -> length
    draw(img, hits, out)                                                   overlay for the answer
"""
import re, json
import numpy as np, cv2

_OCR = None


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def read_text(img, tile=1600, step=1300):
    """img: HxWx3 BGR. Returns [{'text', 'box': [x0,y0,x1,y1], 'conf'}] for the whole sheet."""
    H, W = img.shape[:2]; raw = []
    for y in range(0, max(1, H - tile + step), step):
        for x in range(0, max(1, W - tile + step), step):
            r, _ = _ocr()(img[y:y + tile, x:x + tile])
            for box, txt, conf in r or []:
                xs = [p[0] + x for p in box]; ys = [p[1] + y for p in box]
                raw.append({"text": txt, "box": [min(xs), min(ys), max(xs), max(ys)], "conf": float(conf)})
    out = []                                              # the same text seen in two overlapping tiles counts once
    for r in sorted(raw, key=lambda r: -r["conf"]):
        cx, cy = (r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2
        if any(k["text"] == r["text"] and abs((k["box"][0] + k["box"][2]) / 2 - cx) < 25 and abs((k["box"][1] + k["box"][3]) / 2 - cy) < 25 for k in out):
            continue
        out.append(r)
    return out


def count_tags(texts, prefix):
    """Tags like 'CV-3' / 'PS 12'. distinct = different numbers (the number of tagged objects), instances = every label."""
    pat = re.compile(r"(?<![A-Z])" + re.escape(prefix.upper()) + r"\s?-\s?(\d+)\b")
    hits = []
    for t in texts:
        for m in pat.finditer(t["text"].upper()):
            hits.append({"label": f"{prefix.upper()}-{m.group(1)}", "box": t["box"], "num": int(m.group(1))})
    nums = sorted({h["num"] for h in hits})
    return {"distinct": len(nums), "numbers": nums, "instances": len(hits), "hits": hits}


def count_symbol(img, box, thresh=0.75, scales=(0.9, 1.0, 1.1), rotations=True, roi=None, verify_text=None):
    """Find every copy of the symbol inside `box` (x0,y0,x1,y1) by normalised cross-correlation, incl. 90° rotations and mirror images."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    x0, y0, x1, y1 = [int(v) for v in box]
    tpl = g[y0:y1, x0:x1]
    search = g if roi is None else g[roi[1]:roi[3], roi[0]:roi[2]]
    ox, oy = (0, 0) if roi is None else (roi[0], roi[1])
    variants = [tpl]
    if rotations:
        variants = [np.rot90(tpl, k) for k in range(4)] + [np.rot90(cv2.flip(tpl, 1), k) for k in range(4)]
    cands = []
    for v in variants:
        for s in scales:
            t = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else v
            if t.shape[0] < 6 or t.shape[1] < 6 or t.shape[0] >= search.shape[0] or t.shape[1] >= search.shape[1]: continue
            if t.std() < 1: continue
            res = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= thresh)
            for y, x in zip(ys, xs):
                cands.append((float(res[y, x]), x + ox, y + oy, x + ox + t.shape[1], y + oy + t.shape[0]))
    cands.sort(key=lambda c: -c[0]); keep = []
    for c in cands:                                        # one hit per location
        if any(_iou(c[1:], k[1:]) > 0.3 for k in keep): continue
        keep.append(c)
    hits = [{"box": list(k[1:]), "score": round(k[0], 3)} for k in keep]
    if verify_text:                                        # symbols that carry text (e.g. 'PG' in a circle): read each candidate
        ok = []
        for h in hits:
            x0_, y0_, x1_, y1_ = [int(v) for v in h["box"]]
            for p_ in (0, int(0.3 * max(x1_ - x0_, y1_ - y0_))):   # exact crop first, then with a margin in case the text was cut
                crop = cv2.resize(img[max(0, y0_ - p_):y1_ + p_, max(0, x0_ - p_):x1_ + p_], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                r, _ = _ocr()(crop)
                txt = " ".join(t for _, t, _ in (r or [])).upper().replace(" ", "")
                if verify_text.upper() in txt: ok.append(dict(h, label=verify_text.upper())); break
        hits = ok
    return {"count": len(hits), "hits": hits}


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def scale_from_bar(img, rough, box=None):
    from scale_bar import refine_scale
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return refine_scale(g, rough, box)


def region_area(img, seed, ppu, close=3):
    """Area enclosed by drawn lines around `seed` (x, y), in units² (ppu = pixels per unit). Small gaps (doorways) are closed first."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ink = (g < 160).astype(np.uint8)
    if close: ink = cv2.dilate(ink, np.ones((close, close), np.uint8))
    m = np.zeros((g.shape[0] + 2, g.shape[1] + 2), np.uint8)
    free = (1 - ink) * 255
    cv2.floodFill(free, m, (int(seed[0]), int(seed[1])), 128)
    reg = (free == 128).astype(np.uint8)
    if close: reg = cv2.dilate(reg, np.ones((close, close), np.uint8))   # give back what the closing took
    cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True).reshape(-1, 2).tolist() if cs else []
    return {"area": float(reg.sum()) / ppu ** 2, "polygon": poly}


def colour_area(img, rgb, ppu, roi=None, tol=14, fill=21):
    """Area of everything painted in a legend colour (RGB), with furniture/line work inside those regions filled in."""
    from scipy import ndimage
    a = img[..., ::-1].astype(int)
    m = (np.abs(a - np.array(rgb)).max(2) <= tol)
    if roi is not None:
        r = np.zeros_like(m); r[roi[1]:roi[3], roi[0]:roi[2]] = True; m &= r
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((fill, fill), np.uint8)).astype(bool)
    m = ndimage.binary_fill_holes(m)
    return {"area": float(m.sum()) / ppu ** 2, "mask": m}


def polyline_length(points, ppu):
    p = np.asarray(points, float)
    return float(np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1)).sum()) / ppu


def draw(img, hits, out, colour=(0, 0, 255), label=None, width=4):
    o = img.copy()
    for h in hits:
        x0, y0, x1, y1 = [int(v) for v in h["box"]]
        cv2.rectangle(o, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), colour, width)
        if h.get("label"): cv2.putText(o, h["label"], (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    if label:
        cv2.rectangle(o, (10, 10), (30 + 22 * len(label), 70), (255, 255, 255), -1)
        cv2.putText(o, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)
    cv2.imwrite(out, o)
    return out
