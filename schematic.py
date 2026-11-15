"""Sheet -> schematic: everything on a drawing that can be read deterministically, extracted ONCE per sheet and cached.
Questions are then answered from the schematic (no model call) when they can be; otherwise ask.py's planner takes over.

    python3 schematic.py --image examples/pdf300_bsc-molio_molio-with-uris_plan-05-2-sal.png --out schematic_molio
      -> schematic_molio.json (scale, texts, tags, rooms with areas) + schematic_molio.jpg (everything drawn on the sheet)
    python3 schematic.py --image ... -q "What is the net floor area of the room 'Gang (Rum nr. 201)'?"

What is in it (each item keeps its pixel location, so every answer can be shown on the drawing):
  scale  : px per metre, from the scale bar measured in pixels + the numbers printed on it (OCR)
  texts  : every piece of text (OCR), with boxes
  tags   : tagged objects (PS-12, FSCV-3 ...) grouped by prefix
  rooms  : rooms painted in a fill colour (BIM/CAD exports): polygon, area in m2, the labels inside it
"""
import argparse, json, math, os, re, time
import numpy as np, cv2

UNIT = {"m": 1.0, "mm": 0.001, "cm": 0.01, "ft": 0.3048, "'": 0.3048, "in": 0.0254}
_CACHE = {}


def _texts(img):
    import ask
    return ask.sheet_text(img)


def read_scale(img, texts):
    """Find a scale bar (any style) and read its numbers; px/m = bar length in px / largest number on it."""
    from scale_bar import find_bars, segmented_bars
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); H, W = g.shape
    n, cc, st, _ = cv2.connectedComponentsWithStats((g < 128).astype(np.uint8), 8)
    cands = segmented_bars(st[1:], 0, 0, W, H) + find_bars(g) + find_bars(g, relaxed=True)
    best = None; fallback = None
    for x0, x1, y, L, _f in cands[:60]:
        if L < 40: continue
        if L > 0.4 * max(W, H): continue                    # frame and title-block lines: a scale bar is a small part of the sheet
        near = [t for t in texts if x0 - 0.15 * L <= (t["box"][0] + t["box"][2]) / 2 <= x1 + 0.25 * L
                and abs((t["box"][1] + t["box"][3]) / 2 - y) <= max(40, 0.25 * L)]
        if not any(re.fullmatch(r"\d+(?:[.,]\d+)?\s*(mm|cm|m|ft|')?", t["text"].strip().lower()) for t in near):
            # small tick numbers are often missed on the whole sheet: read the bar's surroundings again, enlarged 3x
            import drawing_tools as T
            cx0, cx1 = int(max(0, x0 - 0.15 * L)), int(min(W, x1 + 0.25 * L)); pad = int(max(40, 0.25 * L))
            cy0, cy1 = int(max(0, y - pad)), int(min(H, y + pad))
            crop = cv2.resize(img[cy0:cy1, cx0:cx1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            near = [{"text": t["text"], "box": [cx0 + t["box"][0] / 3, cy0 + t["box"][1] / 3, cx0 + t["box"][2] / 3, cy0 + t["box"][3] / 3]}
                    for t in T.read_text(crop)]
        nums, unit = [], None
        for t in near:
            s = t["text"].strip().lower().replace(",", ".")
            m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mm|cm|m|ft|')?", s)
            if m:
                nums.append(float(m.group(1)))
                unit = unit or m.group(2)
        if not nums or max(nums) <= 0:
            # tick numbers unreadable: use the printed ratio and paper size ('1:100', 'A3') for a rough scale, then
            # snap the bar to the nearest round length (bars are 1, 2, 5, 10 ... m) for the exact one
            rough = _ratio_scale(texts, W, H)
            if rough:
                m_ = L / rough; k_ = 10 ** math.floor(math.log10(m_))
                nice = min((n_ * k_ * f_ for n_ in (1, 2, 2.5, 3, 4, 5, 6, 8) for f_ in (1, 10)), key=lambda v: abs(m_ / v - 1))
                dev = abs(m_ / nice - 1)
                if dev < 0.12 and (fallback is None or dev < fallback[0]):     # the bar that fits a round length best
                    fallback = (dev, dict(px_per_m=L / nice, bar_px=L, label=f"{nice:g} m (with {rough_text(texts)})",
                                          box=[int(x0), int(y) - 15, int(x1), int(y) + 15], support=1))
            continue
        support = len(nums) + (2 if unit else 0) + (1 if 0.0 in nums else 0)
        metres = max(nums) * UNIT.get(unit or "m", 1.0)
        c = dict(px_per_m=L / metres, bar_px=L, label=f"{max(nums):g} {unit or 'm'}", box=[int(x0), int(y) - 15, int(x1), int(y) + 15],
                 support=support)
        # plausibility: a whole sheet covers metres to hundreds of metres. A long line read with a small number next to
        # it (a dimension, a note) gives a sheet a few metres wide: not the scale bar
        extent = max(W, H) / c["px_per_m"]
        if not (4 <= extent <= 3000): continue
        rough = _ratio_scale(texts, W, H)                    # a printed ratio + paper size, when the sheet has both
        if rough and abs(c["px_per_m"] / rough - 1) > 0.35: continue
        if best is None or support > best["support"]: best = c
    return best or (fallback[1] if fallback else None)


PAPER_MM = {"A0": 1189, "A1": 841, "A2": 594, "A3": 420, "A4": 297}   # long side


def rough_text(texts):
    r = [t["text"] for t in texts if re.search(r"1\s*:\s*\d{2,4}", t["text"])]
    return r[0] if r else ""


def _ratio_scale(texts, W, H):
    """px per m from a printed ratio + paper size on the sheet, assuming the image is the whole sheet."""
    ratio = paper = None
    for t in texts:
        m = re.search(r"1\s*:\s*(\d{2,4})", t["text"]); ratio = ratio or (int(m.group(1)) if m else None)
        p = re.search(r"\b(A[0-4])\b", t["text"]); paper = paper or (p.group(1) if p else None)
    if not ratio or not paper: return None
    px_per_mm_paper = max(W, H) / PAPER_MM[paper]
    return px_per_mm_paper * 1000 / ratio


def read_tags(texts):
    """Tagged objects: PREFIX-number tokens (FSCV-3, PS 12). distinct numbers per prefix = number of tagged objects."""
    tags = {}
    for t in texts:
        for m in re.finditer(r"(?<![A-Z0-9])([A-Z]{1,5})\s?-\s?(\d{1,4})(?![\d.])", t["text"].upper()):
            d = tags.setdefault(m.group(1), {"numbers": set(), "boxes": []})
            d["numbers"].add(int(m.group(2))); d["boxes"].append([int(v) for v in t["box"]])
    return {k: {"count": len(v["numbers"]), "numbers": sorted(v["numbers"]), "boxes": v["boxes"]} for k, v in tags.items()
            if len(v["numbers"]) >= 2}


def read_rooms(img, texts, ppm):
    """Rooms painted in a flat fill colour (how BIM/CAD exports draw spaces). One pass per fill colour:
    connected areas of that colour, text/symbol holes under 1 m2 counted as room, half of each boundary line added."""
    from scipy import ndimage
    if not ppm: return []
    a = img.astype(np.int16); H, W = a.shape[:2]
    mn, mx = a.min(2), a.max(2)
    cand = (mn > 170) & (mn < 250)                                  # light, but not paper white
    if cand.sum() < ppm ** 2: return []
    q = (a[cand] // 6).astype(np.int32); key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts)[:8]
    rooms, taken = [], np.zeros((H, W), bool)
    # A painted space reaches the middle of the line that outlines it (the fill is painted under the stroke).
    # Stroke width = most common short dark run on the sheet; grow each room by the anti-aliased pixel + half a stroke.
    dark = (mn < 100).astype(np.int8)
    d = np.diff(np.pad(dark, ((0, 0), (1, 1))), axis=1)
    st_, en_ = np.where(d == 1), np.where(d == -1)
    runs = (en_[1] - st_[1]); runs = runs[(runs >= 1) & (runs <= 10)]
    stroke = int(np.bincount(runs).argmax()) if len(runs) else 2
    grow = 1 + int(round(stroke / 2))
    for k in order:
        if counts[k] < 0.5 * ppm ** 2: break
        c = np.array([vals[k] // 10000, (vals[k] // 100) % 100, vals[k] % 100]) * 6 + 3
        m = (np.abs(a - c).max(2) <= 8) & ~taken
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
        has_text = np.zeros(n, bool)
        for t in texts:
            cy_, cx_ = min(H - 1, int((t["box"][1] + t["box"][3]) / 2)), min(W - 1, int((t["box"][0] + t["box"][2]) / 2))
            has_text[lab[cy_, cx_]] = True
        # pieces of a room cut off by thin drawn lines (door swings, furniture outlines): small, unlabelled, same colour,
        # separated from exactly one bigger room piece by a line a few pixels wide -> they belong to that room
        owner = {}
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area >= 1.5 * ppm ** 2 or has_text[i] or area < 0.02 * ppm ** 2: continue
            y0_, x0_ = max(0, y - 8), max(0, x - 8)
            sub = lab[y0_:y + h + 8, x0_:x + w + 8]
            grown = cv2.dilate((sub == i).astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
            nb = [j for j in np.unique(sub[grown]) if j not in (0, i) and st[j][4] >= 0.5 * ppm ** 2]
            if len(nb) == 1: owner[i] = nb[0]
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if i in owner or area < 0.5 * ppm ** 2 or area > 0.4 * H * W: continue
            parts = [i] + [k for k, v in owner.items() if v == i]
            xs0 = min(st[k][0] for k in parts); ys0 = min(st[k][1] for k in parts)
            xs1 = max(st[k][0] + st[k][2] for k in parts); ys1 = max(st[k][1] + st[k][3] for k in parts)
            x, y, w, h = xs0, ys0, xs1 - xs0, ys1 - ys0
            reg = np.isin(lab[y:y + h, x:x + w], parts)
            if len(parts) > 1:                               # the thin line between the pieces belongs to the room too
                reg = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool) | reg
            holes = ndimage.binary_fill_holes(reg) & ~reg
            hl, nh = ndimage.label(holes)
            if nh:
                sizes = ndimage.sum(holes, hl, range(1, nh + 1))
                reg |= np.isin(hl, [j + 1 for j, s in enumerate(sizes) if s / ppm ** 2 < 1.0])
            pad = grow + 1
            reg = np.pad(reg, pad); x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
            reg = cv2.dilate(reg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * grow + 1, 2 * grow + 1))).astype(bool)
            x0c, y0c = max(0, -x), max(0, -y); reg = reg[y0c:, x0c:]; x, y = x + x0c, y + y0c
            reg = reg[:H - y, :W - x]; h, w = reg.shape
            full = np.zeros((H, W), bool); full[y:y + h, x:x + w] = reg
            taken |= full
            cs, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly = max(cs, key=cv2.contourArea).reshape(-1, 2).tolist() if cs else []
            labels = [t for t in texts if full[min(H - 1, int((t["box"][1] + t["box"][3]) / 2)), min(W - 1, int((t["box"][0] + t["box"][2]) / 2))]]
            labels.sort(key=lambda t: (t["box"][1], t["box"][0]))
            rooms.append(dict(area_m2=round(float(full.sum() / ppm ** 2), 2), polygon=poly, bbox=[int(x), int(y), int(x + w), int(y + h)],
                              labels=[t["text"] for t in labels], label_boxes=[[int(v) for v in t["box"]] for t in labels],
                              pieces=len(parts), grown_px=grow))
    return rooms


def read_rooms_white(img, texts, ppm):
    """Rooms on plans that are not painted (scans, CubiCasa, Swiss): white space enclosed by walls.
    Door openings are closed first (a gap up to ~1 m in a wall line), then each enclosed white area that carries a
    text label is a room. Areas only when the scale is known."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); H, W = g.shape
    ink = (g < 160).astype(np.uint8)
    # walls: solid, grey or hatched wall bodies (anything clearly not paper, hatching closed up) that survive an
    # opening wider than text strokes and thin drawing lines
    body = cv2.morphologyEx((g < 215).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    t = max(3, int(round((0.06 * ppm) if ppm else 0.004 * max(H, W))))
    walls = cv2.morphologyEx(body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (t, t)))
    if walls.sum() < 0.002 * H * W:                          # thin-line drawing: all ink is the boundary
        walls = ink
    scale_px = ppm
    if not scale_px:                                         # no scale: estimate from wall thickness (walls ~0.25 m)
        dt = cv2.distanceTransform(walls, cv2.DIST_L2, 3)
        v = dt[dt > 0]
        scale_px = (4 * float(np.median(v)) / 0.25) if len(v) else 0.004 * max(H, W) / 0.06
    L = int(round(1.1 * scale_px))                           # widest door opening to bridge
    closed = walls.copy()
    for k in ((1, L), (L, 1)):
        closed |= cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    free = (closed == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    px_m2 = scale_px ** 2
    rooms = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H: continue          # the outside
        if area < 1.0 * px_m2 or area > 300 * px_m2: continue
        reg = lab[y:y + h, x:x + w] == i
        labels = [t_ for t_ in texts if x <= (t_["box"][0] + t_["box"][2]) / 2 < x + w and y <= (t_["box"][1] + t_["box"][3]) / 2 < y + h
                  and reg[int((t_["box"][1] + t_["box"][3]) / 2) - y, int((t_["box"][0] + t_["box"][2]) / 2) - x]]
        if not labels: continue
        cs, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = (max(cs, key=cv2.contourArea).reshape(-1, 2) + [x, y]).tolist() if cs else []
        labels.sort(key=lambda t_: (t_["box"][1], t_["box"][0]))
        rooms.append(dict(area_m2=round(float(area / ppm ** 2), 2) if ppm else None, polygon=poly, bbox=[int(x), int(y), int(x + w), int(y + h)],
                          labels=[t_["text"] for t_ in labels], label_boxes=[[int(v) for v in t_["box"]] for t_ in labels], pieces=1, white=True))
    return rooms


def build(img, key=None):
    if key and key in _CACHE: return _CACHE[key]
    t0 = time.time()
    texts = _texts(img)
    scale = read_scale(img, texts)
    ppm = scale["px_per_m"] if scale else None
    # a real room carries one label block (name + number, maybe an area); a region holding many labels is a building
    # outline or background painted in the room colour, not a room
    rooms = [r for r in read_rooms(img, texts, ppm) if r["labels"] and len(r["labels"]) <= 5]
    # Painted rooms count only when the sheet is actually painted room by room: they must hold most of the printed
    # room numbers (3.10, 18-2 ...). A plan that only greys its stair and balcony is not a painted plan.
    numbered = [t for t in texts if re.fullmatch(r"\d{1,3}[.\-]\d{1,3}[A-Za-z]?", t["text"].strip())]
    inside = sum(1 for r in rooms for l in r["labels"] if re.fullmatch(r"\d{1,3}[.\-]\d{1,3}[A-Za-z]?", l.strip()))
    painted = len(rooms) >= 2 and (len(numbered) < 3 or inside >= 0.5 * len(numbered))
    rooms_source = "painted" if painted else "none"
    if not painted: rooms = []
    sch = dict(image_size=list(img.shape[1::-1]), scale=scale, n_texts=len(texts), rooms_source=rooms_source,
               texts=[{"text": t["text"], "box": [int(v) for v in t["box"]]} for t in texts],
               tags=read_tags(texts), rooms=rooms, seconds=round(time.time() - t0, 1))
    if key: _CACHE[key] = sch
    return sch


# ---------- answering from the schematic alone (no model call) ----------

def _room_named(sch, name):
    import ask
    hits = ask.find_label([{"text": " ".join(r["labels"]), "box": r["bbox"]} for r in sch["rooms"]], name)
    if len(hits) == 1:
        return next(r for r in sch["rooms"] if r["bbox"] == hits[0]["box"])
    for r in sch["rooms"]:                                  # the number alone, e.g. '201' or '18-2'
        nums = re.findall(r"\d+(?:[.\-/]\d+)*", name)
        if nums and any(re.search(r"(?<![\d.\-])" + re.escape(nums[-1]) + r"(?![\d])", l) for l in r["labels"]):
            same = [o for o in sch["rooms"] if any(re.search(r"(?<![\d.\-])" + re.escape(nums[-1]) + r"(?![\d])", l) for l in o["labels"])]
            if len(same) == 1: return r
    return None


def _only_this_room(r, name):
    """The room must not also carry another room's name: a corridor holding a shaft's label is not the shaft."""
    want = {w for w in re.findall(r"[^\W\d_]{3,}", name.upper())}
    for l in r["labels"]:
        words = [w for w in re.findall(r"[^\W\d_]{3,}", l.upper()) if w not in {"RUM", "ROOM", "RAUM", "NR"}]
        if words and not any(w in want or any(w in x or x in w for x in want) for w in words):
            return False
    return True


def answer(sch, q):
    """(value, evidence) for questions the schematic answers on its own, else None (the planner takes over)."""
    m = re.search(r"[\"'„“«‘]([^\"'”“»’]+)[\"'”“»’]", q)        # quoted room name: "net floor area of the room 'Gang (Rum nr. 201)'"
    if m and re.search(r"area|fläche|areal|oppervlak|pind|superficie|pinta|m²|m2", q, re.I):
        r = _room_named(sch, m.group(1))
        if r and _only_this_room(r, m.group(1)): return r["area_m2"], {"kind": "room", "room": r}
    m = re.search(r"\(([A-Z]{1,5})\)|\b([A-Z]{2,5})-numbers\b", q)   # '(FSCV)', 'PS-numbers'
    if m and re.search(r"how many|count|number of", q, re.I):
        p = m.group(1) or m.group(2)
        if p in sch["tags"]: return sch["tags"][p]["count"], {"kind": "tags", "prefix": p, "tags": sch["tags"][p]}
    return None


def draw(img, sch, out, highlight=None):
    o = img.copy(); H, W = o.shape[:2]; fs = max(0.6, W / 4000)
    rng = np.random.default_rng(0)
    for r in sch["rooms"]:
        col = tuple(int(v) for v in rng.integers(60, 220, 3))
        cv2.polylines(o, [np.int32(r["polygon"])], True, col, max(2, int(3 * fs)))
        x0, y0, x1, y1 = r["bbox"]
        if r["area_m2"] is not None:
            cv2.putText(o, f"{r['area_m2']:.2f} m2", (x0 + 8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fs, col, max(1, int(2 * fs)))
    for p, d in sch["tags"].items():
        for b in d["boxes"]: cv2.rectangle(o, tuple(b[:2]), tuple(b[2:]), (200, 0, 200), max(1, int(2 * fs)))
    if sch["scale"]:
        b = sch["scale"]["box"]; cv2.rectangle(o, tuple(b[:2]), tuple(b[2:]), (0, 0, 230), max(2, int(3 * fs)))
        cv2.putText(o, f"scale {sch['scale']['label']} = {sch['scale']['bar_px']} px", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 230), 2)
    if highlight and highlight.get("kind") == "room":
        r = highlight["room"]; m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.int32(r["polygon"])], 1)
        for o_ in sch["rooms"]:                              # rooms enclosed by this one are not part of it
            if o_ is not r: cv2.fillPoly(m, [np.int32(o_["polygon"])], 0)
        m = m.astype(bool)
        o[m] = (0.55 * o[m] + 0.45 * np.array([255, 140, 0])).astype(np.uint8)
    if highlight and highlight.get("kind") == "tags":
        for b in highlight["tags"]["boxes"]: cv2.rectangle(o, (b[0] - 6, b[1] - 6), (b[2] + 6, b[3] + 6), (0, 0, 230), max(3, int(5 * fs)))
    cv2.imwrite(out, o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); ap.add_argument("--out", default="")
    ap.add_argument("-q", "--question", default="")
    a = ap.parse_args()
    img = cv2.imread(a.image); out = a.out or "schematic_" + os.path.splitext(os.path.basename(a.image))[0]
    sch = build(img)
    hl = None
    if a.question:
        res = answer(sch, a.question)
        print(json.dumps({"question": a.question, "answer": res[0] if res else None,
                          "source": "schematic (no model call)" if res else "not answerable from the schematic: use ask.py"}, ensure_ascii=False))
        hl = res[1] if res else None
    json.dump(sch, open(out + ".json", "w"), ensure_ascii=False, default=str)
    draw(img, sch, out + ".jpg", hl)
    print(f"scale: {sch['scale']['label'] if sch['scale'] else None} ({round(sch['scale']['px_per_m'], 1) if sch['scale'] else None} px/m) | "
          f"{sch['n_texts']} texts | tag groups {', '.join(f'{k}:{v['count']}' for k, v in sch['tags'].items())} | "
          f"{len(sch['rooms'])} labelled rooms | {sch['seconds']} s -> {out}.json/.jpg")
