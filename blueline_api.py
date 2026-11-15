"""Blueline API: our deterministic reader behind the frontend's contract (github.com/Paulsg777/blueline-stoa).

    python3 blueline_api.py                 -> http://localhost:8000   (frontend: Reader -> Blueline API -> that URL)

GET  /health                         -> 200 {"ok": true, ...}
POST /analyze  multipart: file (PNG/JPG), sheet (JSON: id, building, level)
               -> BluelineSchema, read deterministically (no model call): scale from the scale bar, painted rooms with
                  polygons and measured areas, all text, tagged objects; doors/windows/fixtures from our detectors when
                  their weights are present (FP_YOLO / yolo_cubi.pt). Extra fields the frontend may use:
                  rooms[].measuredAreaM2 (net area measured on the drawing; holes such as enclosed shafts excluded),
                  rooms[].labelBox, tags[], meta.msRead / costEUR (0: no model call).
POST /ask      multipart: file, question  -> our full pipeline for questions a schema cannot answer
               (schematic first, then Gemini-planned local tools, then the model as a last resort):
               {answer, unit, status: "confident" | "check", source, modelCalls, tokens, methods[], evidence{boxes, polygons}}
Only the standard library is needed for the server itself. CORS is open (*) so a page opened from file:// can call it.
"""
import io, json, os, re, sys, time, traceback
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ask, schematic, arcs

READER_VERSION = "perceive+measure+check v16"     # part of the cache key: a new reader version never reuses old reads
CACHE_DIR = os.environ.get("READ_CACHE", "read_cache")

# room label words (several languages) -> the frontend's room types
TYPE_WORDS = [
    ("WC", r"\bWC\b|TOILET|TUALET|VESSA"), ("Bathroom", r"BATH|BAIN|\bKPH\b|\bPH\b|BAGNO|BAD\b|BADEZ|VANNI|DUSCH"),
    ("Sauna", r"SAUNA|\bS\b|LEIL"), ("Kitchen", r"KITCHEN|CUISINE|CUCIN|CUDIN|KEITTI|KÖÖK|KOOK|KÜCHE|KUCHE|CUCINA|\bK\b|\bKK\b|TEKØKKEN|TEKOKKEN|KEUKEN"),
    ("Living", r"LIVING|SÉJOUR|SEJOUR|REPAS|\bOH\b|OLOH|ELUTUBA|SOGGIORNO|WOHN|STUE|LOUNGE|WOONK"),
    ("Bedroom", r"BED|PIÈCE|PIECE|CHAMBRE|\bMH\b|MAKUU|MAGAMIS|SCHLAF|CAMERA|STANZA|ZIMMER|SOVE|SLAAPK|\bH\b|HAB"),
    ("Hall", r"\bHALL|\bET\b|ETEINEN|ESIK|ENTR|INGRESSO|DIELE|EINGANG|ENTRÉ"), ("Corridor", r"CORRIDOR|COULOIR|GANG|KORIDOR|FLUR|KÄYTÄVÄ|CORRIDOIO|LOBBY"),
    ("Storage", r"STOR|\bVH\b|VARASTO|DEPOT|ABST|PANIEKA|BERGING|ABIRUUM|KAST|LAGER|RIPOSTIGLIO"),
    ("Technical", r"TECH|TEKNIK|SCHACHT|SKAKT|SHAFT|ELEKTR|\bLVI\b|IV-|SERVER"), ("Stair", r"STAIR|ESCALIER|TREPP|TRAPP|PORRA|SCALA|TREPI"),
    ("Office", r"OFFICE|KONTOR|BÜRO|BURO|TYÖH|UFFICIO|KANTOOR"), ("Balcony", r"BALCON|BALKON|PARVEKE|RÕDU|LOGGIA|TERRA"),
    ("Utility", r"UTILITY|KODINH|\bKHH\b|PESU|WASH|BUANDERIE"), ("Meeting", r"MEETING|MØDE|MODE|NEUVOTT|BESPREK"),
]


def room_type(labels):
    t = " ".join(labels).upper()
    for typ, pat in TYPE_WORDS:
        if re.search(pat, t): return typ
    return "Room"


# ---------- perception: one model call per sheet says WHAT is there; deterministic tools measure and verify ----------
PERCEIVE_PROMPT = """You are reading a construction drawing sheet (an architectural floor plan, or a piping / mechanical / electrical /
structural / site sheet). List what is there; do not measure anything.
Coordinates: a 0-1000 grid over the WHOLE image (x from left, y from top).
Return JSON only:
{"sheet_type": "floor_plan" (architectural plan of rooms) | "piping" | "mechanical" | "electrical" | "structural" | "site" | "detail" | "other",
 "rooms": [{"name": "room name exactly as printed, or a short description if unlabelled", "number": "room number exactly as printed or null",
            "point": [x, y] a point well inside the room, on the room's label if it has one,
            "area": the room's area as printed on the sheet, as a number in m² (e.g. 17,0 m² -> 17.0), or null if none is printed}],
 "items": [{"kind": "door"|"window"|"toilet"|"sink"|"kitchen_sink"|"shower"|"bathtub"|"stove"|"sauna_stove", "point": [x, y] its centre}],
 "symbols": [{"kind": "what it is, in plain lower-case English, e.g. check valve, gate valve, pressure relief valve, pump, flow meter, pressure gauge, filter",
              "tag": "the identifier printed next to it exactly as printed (e.g. CV-3, PS-12) or null", "point": [x, y] its centre}],
 "scale_bar": {"box": [x0,y0,x1,y1], "length": the largest number printed on the bar, "unit": "m"/"mm"/"ft"} or null,
 "building_box": [x0, y0, x1, y1] a tight box around the building's outer walls (balconies included; not the title block,
                 legend, notes, frame or scale bar)}
Rooms: only on a floor_plan sheet; on any other sheet type return "rooms": []. On a floor plan include every room, corridor,
bathroom, balcony, storage and shaft drawn inside the building outline; exclude the title block, legend, notes and anything outside.
Items: list EVERY door (each door leaf or doorway drawn, including balcony and entrance doors), every window (each window
opening in the outer walls), every toilet bowl, washbasin ("sink"), kitchen sink ("kitchen_sink"), shower, bathtub and stove,
one entry per physical object, each with its own point.
Symbols: every other piece of equipment or fitting drawn as a symbol (valves, pumps, meters, gauges, filters, strainers,
tanks, outlets ...), one entry per physical object, named by what it is (use the sheet's legend and notes when they say).
Not text-only notes, not pipe labels, not section or detail markers."""
_PERC = {}


def perceive(img, model=None):
    model = model or os.environ.get("PERCEIVE_MODEL", "gemini-4-argon")
    key = (ask._img_key(img), model)                       # a different perception model is a different read
    if key in _PERC: return _PERC[key]
    from google import genai
    from google.genai import types
    H, W = img.shape[:2]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last = None
    for side in (3072, 2048, 1600):                         # retry smaller when the upload is cut off
        s = min(1.0, side / max(H, W))
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        try:
            r = client.models.generate_content(model=model,
                                               contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"), PERCEIVE_PROMPT],
                                               config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
            break
        except Exception as e:
            if type(e).__name__ == "ClientError":            # 4xx: wrong model name, no access, bad option; a smaller image won't help
                raise RuntimeError(f"{model}: {str(e)[:300]}") from e
            last = e; print(f"perception call ({model}) failed at {side} px ({type(e).__name__}); retrying smaller", flush=True)
    else:
        raise last
    p = json.loads(r.text); u = getattr(r, "usage_metadata", None)
    try:                                                    # keep what the model saw, to inspect and to reuse
        os.makedirs("perception_cache", exist_ok=True)
        json.dump(dict(p, _image_size=[W, H], _model=model), open(os.path.join("perception_cache", f"{abs(hash(key)) % 10**10}.json"), "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass
    p["_tokens"] = sum((getattr(u, k, 0) or 0) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")) if u else 0
    _PERC[key] = p
    return p


def split_along_walls(img, m, x0, y0, pts, ppm, texts=()):
    """Split one region holding several rooms: each room grows from its label over free floor (watershed on the distance
    to the nearest drawn line), so the boundaries fall on walls and at the narrowest passages (doorways), not on straight
    lines between labels. Returns an index map (which room each pixel of the region belongs to)."""
    h, w = m.shape
    crop = img[y0:y0 + h, x0:x0 + w]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).copy()
    for t in texts:                                         # printed text is not a wall: erase it from the floor
        bx0, by0, bx1, by1 = [int(v) for v in t["box"]]
        g[max(0, by0 - y0 - 2):max(0, by1 - y0 + 2), max(0, bx0 - x0 - 2):max(0, bx1 - x0 + 2)] = 255
    free = ((g > 170) & (m > 0)).astype(np.uint8)
    # flat relief: every room floods outward at the same speed over free floor only, so a pixel goes to the room it is
    # nearest to WALKING (around walls, through doorways), not as the crow flies
    relief = np.where(free > 0, 0, 255).astype(np.uint8)
    markers = np.zeros((h, w), np.int32); markers[m == 0] = 1
    r = max(3, int(0.12 * ppm) if ppm else 6)
    for j, (px, py) in enumerate(pts):
        cx, cy = int(px - x0), int(py - y0)
        if not (0 <= cx < w and 0 <= cy < h): continue
        if not free[cy, cx]:                                # the label sits on ink: start from the nearest free pixel
            ys, xs = np.where(free[max(0, cy - 4 * r):cy + 4 * r, max(0, cx - 4 * r):cx + 4 * r])
            if len(xs):
                k = np.argmin((ys - min(cy, 4 * r)) ** 2 + (xs - min(cx, 4 * r)) ** 2)
                cx, cy = max(0, cx - 4 * r) + xs[k], max(0, cy - 4 * r) + ys[k]
        cv2.circle(markers, (int(cx), int(cy)), r, j + 2, -1)
        markers[m == 0] = 1
    cv2.watershed(cv2.merge([relief] * 3), markers)
    who = markers - 2
    bad = (who < 0) & (m > 0)                               # watershed lines: give them to the neighbouring room
    if bad.any():
        grown = cv2.dilate(np.where(who >= 0, who + 1, 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(np.int32) - 1
        who = np.where(bad, grown, who)
    return who


def rooms_from_interior(img, sch, perc, ppm):
    """Rooms without any trained segmenter, for drawing styles it does not know (scans, CubiCasa-style flats):
    the building interior is all floor enclosed by drawn lines (not reachable from the sheet's edge); each room the model
    listed grows from its label over that floor until it meets walls, so neighbours meet at walls and doorways."""
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    for t in sch["texts"]:                                  # printed text is not a wall
        x0, y0, x1, y1 = [int(v) for v in t["box"]]; g[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 255
    free = (g > 170).astype(np.uint8)
    pts = []
    for r in perc.get("rooms") or []:
        try: pts.append((float(r["point"][0]) * W / 1000, float(r["point"][1]) * H / 1000, r))
        except Exception: pass

    # everything outside the building's box (drawing frame, title block, legend) is outside, whatever lines it holds.
    # The box comes from the model; without it, from the listed rooms' points with a margin.
    bb = None
    try:
        q = [float(v) for v in (perc.get("building_box") or [])]
        if len(q) == 4 and q[2] > q[0] and q[3] > q[1]: bb = [q[0] * W / 1000, q[1] * H / 1000, q[2] * W / 1000, q[3] * H / 1000]
    except Exception:
        bb = None
    if len(pts) >= 2:                                       # the box always holds every room the model listed
        xs_, ys_ = [p[0] for p in pts], [p[1] for p in pts]
        mx_, my_ = 0.15 * (max(xs_) - min(xs_)) + 0.03 * W, 0.15 * (max(ys_) - min(ys_)) + 0.03 * H
        pb = [min(xs_) - mx_, min(ys_) - my_, max(xs_) + mx_, max(ys_) + my_]
        bb = pb if bb is None else [min(bb[0], pb[0]), min(bb[1], pb[1]), max(bb[2], pb[2]), max(bb[3], pb[3])]
    if bb is not None:
        m_ = int(0.015 * max(H, W))                         # the model's box is approximate: keep a margin
        x0b, y0b = max(0, int(bb[0]) - m_), max(0, int(bb[1]) - m_)
        x1b, y1b = min(W, int(bb[2]) + m_), min(H, int(bb[3]) + m_)
        if (x1b - x0b) * (y1b - y0b) < 0.97 * W * H:        # the box leaves something out: blank it
            keep = np.zeros((H, W), np.uint8); keep[y0b:y1b, x0b:x1b] = 1
            free = np.where(keep > 0, free, 1).astype(np.uint8)

    def enclosed(gap):
        """Free floor the sheet's edge cannot reach, with gaps in the outline up to `gap` px bridged (dotted or broken
        walls, open balconies): the outline only decides inside vs outside, rooms still meet at doorways."""
        ink = 1 - free
        if gap > 1: ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
        ff = (1 - ink).astype(np.uint8); mask = np.zeros((H + 2, W + 2), np.uint8)
        for sx, sy in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
            if ff[sy, sx] == 1: cv2.floodFill(ff, mask, (sx, sy), 2)
        return ((ff != 2) & (free > 0)).astype(np.uint8)

    def inside_share(m):                                    # how many listed rooms sit on enclosed floor
        if not pts: return 0.0
        ok = 0
        for px, py, _ in pts:
            x, y, r_ = int(px), int(py), max(3, int(0.01 * max(H, W)))
            ok += m[max(0, y - r_):y + r_, max(0, x - r_):x + r_].any()
        return ok / len(pts)
    # the smallest bridging that puts (nearly) every listed room inside the building
    best = None
    for f in (0, 0.006, 0.012, 0.02, 0.03):
        m = enclosed(int(f * max(H, W)))
        sh = inside_share(m)
        if best is None or sh > best[0] + 1e-9: best = (sh, m)
        if sh >= 0.85: break
    interior = best[1]
    ctr = lambda t: ((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
    snapped = []
    for px, py, r in pts:                                   # move onto the printed label when OCR finds it nearby
        name = str(r.get("number") or r.get("name") or "").strip()
        hits = [t for t in sch["texts"] if name and t["text"].strip().lower() == name.lower()]
        if hits:
            t = min(hits, key=lambda t: (ctr(t)[0] - px) ** 2 + (ctr(t)[1] - py) ** 2)
            if (ctr(t)[0] - px) ** 2 + (ctr(t)[1] - py) ** 2 < (0.12 * max(H, W)) ** 2: px, py = ctr(t)
        snapped.append((px, py, r, bool(hits)))
    if not snapped or interior.sum() == 0: return []
    # completeness: the model's list can stop early on a large sheet. Every open area of the enclosed floor also gets a
    # seed (centres of open floor), all rooms are split together, then a seed that only split a room in two is merged
    # back: two halves of one room share a long open edge, two rooms only a doorway.
    unit = ppm or (0.012 * max(H, W))
    dtm = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    ys_, xs_ = np.where(dtm > 0.7 * unit)
    auto = []
    for i in np.argsort(-dtm[ys_, xs_])[::5]:
        x, y = int(xs_[i]), int(ys_[i])
        if all((x - a) ** 2 + (y - b) ** 2 > (1.6 * unit) ** 2 for a, b, *_ in snapped) and \
           all((x - a) ** 2 + (y - b) ** 2 > (2.4 * unit) ** 2 for a, b in auto):
            auto.append((x, y))
        if len(snapped) + len(auto) >= 240: break
    # extra seeds only where a listed room swallowed far more floor than a room has (over 3x the median listed room and
    # over ~40 m2): that is a room growing through doorways into rooms the model did not list
    who = split_along_walls(img, interior, 0, 0, [(p[0], p[1]) for p in snapped], ppm, sch["texts"])
    sizes = [int(((who == j) & (interior > 0)).sum()) for j in range(len(snapped))]
    med = float(np.median([z for z in sizes if z > 0])) if any(sizes) else 0
    big = [j for j, z in enumerate(sizes) if med and z > 3 * med and z > 40 * unit ** 2]
    if big:
        zone = np.isin(who, big)
        auto = [(x, y) for x, y in auto if zone[y, x]]
    else:
        auto = []
    seeds = [(p[0], p[1]) for p in snapped] + auto
    if auto: who = split_along_walls(img, interior, 0, 0, seeds, ppm, sch["texts"])
    n0 = len(snapped)
    owner = list(range(len(seeds)))                         # union-find: auto regions merged into a neighbour
    def find(i):
        while owner[i] != i: owner[i] = owner[owner[i]]; i = owner[i]
        return i
    k3 = np.ones((3, 3), np.uint8)
    for j in range(n0, len(seeds)):
        mj = ((who == j) & (interior > 0)).astype(np.uint8)
        if not mj.any(): continue
        ring = (cv2.dilate(mj, k3) > 0) & (mj == 0) & (interior > 0)
        nb = who[ring]; nb = nb[(nb >= 0) & (nb != j)]
        if not len(nb): continue
        ids, cnt = np.unique(nb, return_counts=True)
        best = int(ids[np.argmax(cnt)]); shared = int(cnt.max())
        if shared >= 1.5 * unit:                            # a long open edge: the same room
            a_, b_ = find(j), find(best)
            if a_ != b_: owner[a_] = b_ if b_ < n0 or a_ >= n0 else a_
    for j in range(len(seeds)):
        if find(j) != j: who[who == j] = find(j)
    rooms = []
    for j in range(n0, len(seeds)):                         # unlisted rooms that stayed on their own
        if find(j) != j: continue
        pm = ((who == j) & (interior > 0)).astype(np.uint8)
        if pm.sum() < 2 * unit ** 2: continue               # under ~2 m2: a shaft or a gap, not a room
        cs, _ = cv2.findContours(cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
        rooms.append(dict(id="", type="Room", label="", polygon=poly.tolist(), printedAreaM2=None,
                          measuredAreaM2=round(float(pm.sum()) / ppm ** 2, 2) if ppm else None, conf=0.4, labelVerified=False, source="drawing"))
    unlisted = rooms; rooms = []
    for j, (px, py, r, ok) in enumerate(snapped):
        pm = ((who == j) & (interior > 0)).astype(np.uint8)
        pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        c = max(cs, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        label = " ".join(x for x in (str(r.get("name") or "").strip(), str(r.get("number") or "").strip()) if x)
        rooms.append(dict(id=f"R{len(rooms) + 1}", type=room_type([label]), label=label, polygon=poly.tolist(), printedAreaM2=None,
                          measuredAreaM2=round(float(pm.sum()) / ppm ** 2, 2) if ppm else None, conf=0.8 if ok else 0.5, labelVerified=ok))
    for r in unlisted: r["id"] = f"R{len(rooms) + 1}"; rooms.append(r)
    return rooms


def rooms_from_perception(img, sch, fp, perc, ppm):
    """Each room the model listed -> a measured room. Its label is checked against OCR of the sheet (the point moves onto the
    printed number when OCR finds it); the boundary is the segmenter region holding that point, shared regions are split
    between the rooms they hold; the area is the printed net area near the label when it agrees with the drawn room."""
    H, W = img.shape[:2]
    AREA = re.compile(r"(\d+(?:[.,]\d+)?)\s*m(?:2|²)")
    ctr = lambda t: ((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
    norm = lambda x: re.sub(r"[^0-9a-z]", "", str(x).lower())
    listed = []
    for r in perc.get("rooms") or []:
        try: px, py = float(r["point"][0]) * W / 1000, float(r["point"][1]) * H / 1000
        except Exception: continue
        num = str(r.get("number") or "").strip(); name = str(r.get("name") or "").strip()
        verified = False
        if num:                                             # the printed number, found by OCR -> exact position, verified label
            hits = [t for t in sch["texts"] if norm(num) and re.search(r"(?<![0-9])" + re.escape(num) + r"(?![0-9])", t["text"])]
            if hits:
                t = min(hits, key=lambda t: (ctr(t)[0] - px) ** 2 + (ctr(t)[1] - py) ** 2)
                if (ctr(t)[0] - px) ** 2 + (ctr(t)[1] - py) ** 2 < (0.15 * max(H, W)) ** 2:
                    px, py = ctr(t); verified = True
        if not verified and name:
            hits = [t for t in sch["texts"] if norm(name) and norm(name)[:5] in norm(t["text"])]
            if hits:
                t = min(hits, key=lambda t: (ctr(t)[0] - px) ** 2 + (ctr(t)[1] - py) ** 2)
                if (ctr(t)[0] - px) ** 2 + (ctr(t)[1] - py) ** 2 < (0.08 * max(H, W)) ** 2:
                    px, py = ctr(t); verified = True
        listed.append(dict(name=name, number=num or None, pt=(px, py), verified=verified))
    segs = [np.int32(r["polygon"]) for r in fp["rooms"] if len(r["polygon"]) >= 3]
    owner = {}
    for i, L in enumerate(listed):                          # which region holds each listed room
        inside = [k for k, sp in enumerate(segs) if cv2.pointPolygonTest(sp, L["pt"], False) >= 0]
        if inside: owner.setdefault(min(inside, key=lambda k: abs(cv2.contourArea(segs[k]))), []).append(i)
    rooms = []
    for k, idx in owner.items():
        sp = segs[k]; x0, y0, w, h = cv2.boundingRect(sp)
        m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [sp - [x0, y0]], 1)
        if len(idx) > 1: who = split_along_walls(img, m, x0, y0, [listed[i]["pt"] for i in idx], ppm, sch["texts"])
        for j, i in enumerate(idx):
            pm = (m > 0) if len(idx) == 1 else ((who == j) & (m > 0))
            cs, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs: continue
            poly = max(cs, key=cv2.contourArea).reshape(-1, 2) + [x0, y0]
            L = listed[i]; measured = round(float(pm.sum()) / ppm ** 2, 2) if ppm else None
            txt = [t for t in sch["texts"] if cv2.pointPolygonTest(np.int32(poly), tuple(float(v) for v in ctr(t)), False) >= 0]
            printed = None
            ar = [t for t in txt if AREA.search(t["text"])]
            if ar:
                near = min(ar, key=lambda t: (ctr(t)[0] - L["pt"][0]) ** 2 + (ctr(t)[1] - L["pt"][1]) ** 2)
                printed = float(AREA.search(near["text"]).group(1).replace(",", "."))
                if measured and abs(printed / measured - 1) > 0.25: printed = None
            label = " ".join(x for x in (L["name"], L["number"]) if x)
            rooms.append(dict(id=f"R{len(rooms) + 1}", type=room_type([label]), label=label, polygon=np.int32(poly).tolist(),
                              printedAreaM2=printed, measuredAreaM2=printed if printed else measured,
                              conf=0.9 if L["verified"] else 0.5, labelVerified=L["verified"]))
    return rooms


PLAN_PROMPT = """Turn a question about floor plans into a plan that code will run on the plans' data. Never answer with a number yourself.
Vocabulary of the plans that are loaded (use only these values): {voc}

Plan fields: find ("item" or "room", required); kind (item kind or list, from itemKinds); type (room type or list, from roomTypes);
label (an exact label from labels); in ({{roomType, roomLabel, roomNumber, apartment}} where the item/room must be);
has ({{"kind": ..., "min": 1}} rooms containing items); where ([{{"field": "area", "op": ">"|">="|"<"|"<="|"=", "value": number}}]);
do ("count", "sum_area", "max_area", "min_area"); scope ("floor" = the floor being viewed, "building", "project");
floor (a floor number, 1 = bottom, or "top"; overrides scope); readAs (one plain sentence saying how you read the question).
Exclusions ("all rooms except bathrooms") are written as the list of the remaining types from roomTypes.
A general word ("valves", "pressure valves", "fixtures") means the list of EVERY itemKind it covers, e.g. "pressure valves" ->
every kind with "pressure" in its name that is a valve. Item kinds can be equipment from piping or electrical sheets.
"Total area" / "sum of areas" -> do "sum_area". No floor mentioned -> scope "floor". "In the building" -> scope "building".
Will something fit in rooms (furniture, equipment, a wheelchair turning circle): fit ({{"shape": "rect", "w": number, "l": number,
"unit": "m"|"cm"|"mm"|"ft"|"in"}} or {{"shape": "circle", "d": number, "unit": ...}}, "not": true to find the rooms where it does NOT fit);
code measures it on each room's outline. Use the user's sizes; for accessibility use the standard 1.5 m turning circle and say so in readAs.
Arithmetic on the result (tiles, paint, cost ...): then ([{{"op": "/"|"*"|"+"|"-", "value": number exactly as the user wrote it,
"unit": "m2"|"cm2"|"mm2"|"ft2"|"m"|"cm"|"mm"|"ft"|null}}], applied in order; code converts the units); round ("up"|"down"|"nearest");
resultUnit (plural noun for the result, e.g. "tiles"). Keep the user's numbers and units exactly, even if they look odd.
A question that needs a judgement (is it suitable, compliant, accessible, better; compare floors) is answered from several
measured facts: return {{"plans": [plan, ...] (2-5 plans, each with its own readAs), "conclude": "what to decide from them, one sentence"}}
instead of one plan. Only measurable facts go in plans; the conclusion is written later from their results.
If the question cannot be expressed with these fields, return {{"plan": null}}.

Examples:
{{"plan": {{"find":"item","kind":"door","in":{{"roomType":"Bedroom"}},"do":"count","readAs":"count doors that open into a bedroom, on this floor"}}}}
{{"plan": {{"find":"room","where":[{{"field":"area","op":">","value":15}}],"do":"count","readAs":"count rooms over 15 m², on this floor"}}}}
{{"plan": {{"find":"room","type":["Living","Bedroom","Kitchen","Hall","Corridor"],"do":"sum_area","readAs":"total area of all rooms except bathrooms and WCs, on this floor"}}}}
{{"plan": {{"find":"room","do":"max_area","floor":3,"readAs":"the largest room on floor 3"}}}}
{{"plan": {{"find":"room","type":["Bathroom","WC"],"fit":{{"shape":"circle","d":1.5,"unit":"m","not":true}},"do":"count","readAs":"bathrooms and WCs on this floor where a 1.5 m wheelchair turning circle does not fit (0 = all pass); door widths are not checked"}}}}
{{"plan": {{"find":"room","type":["Bedroom"],"fit":{{"shape":"rect","w":1.6,"l":2.0,"unit":"m"}},"do":"count","readAs":"bedrooms on this floor where a 1.6 x 2.0 m bed fits"}}}}
{{"plan": {{"find":"room","type":["Bathroom"],"do":"sum_area","then":[{{"op":"/","value":0.25,"unit":"m2"}}],"round":"up","resultUnit":"tiles","readAs":"total bathroom floor area on this floor divided by 0.25 m² per tile, rounded up"}}}}

{{"plans": [{{"find":"room","type":["Bathroom","WC"],"fit":{{"shape":"circle","d":1.5,"unit":"m","not":true}},"do":"count","readAs":"bathrooms and WCs where a 1.5 m turning circle does not fit"}},
           {{"find":"room","type":["Bathroom","WC"],"do":"count","readAs":"bathrooms and WCs on this floor"}},
           {{"find":"item","kind":"door","do":"count","readAs":"doors on this floor"}}],
 "conclude": "whether this floor's bathrooms give a wheelchair user room to turn"}}
{{"plans": [{{"find":"room","type":["Bedroom"],"do":"count","readAs":"bedrooms on this floor"}},
           {{"find":"room","type":["Bathroom","WC"],"do":"count","readAs":"bathrooms and WCs on this floor"}},
           {{"find":"room","do":"sum_area","readAs":"total room area on this floor"}}],
 "conclude": "whether this floor suits a family of four"}}

Question: {q}
Return JSON only: {{"plan": {{...}}}}, {{"plans": [...], "conclude": "..."}} or {{"plan": null}}."""


def plan_question(q, voc):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(model=os.environ.get("PLAN_MODEL", "gemini-3.8-flash"),
                                       contents=[PLAN_PROMPT.format(voc=json.dumps(voc, ensure_ascii=False)[:6000], q=q)],
                                       config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    j = json.loads(r.text); u = getattr(r, "usage_metadata", None)
    tok = sum((getattr(u, k, 0) or 0) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")) if u else 0
    rt = set(voc.get("roomTypes") or []); ik = set(voc.get("itemKinds") or [])

    def clean(plan):                                        # keep only vocabulary values: the page must be able to run it
        if not isinstance(plan, dict) or plan.get("find") not in ("item", "room"): return None
        for k_, allowed in (("type", rt), ("kind", ik)):
            if plan.get(k_) is not None and allowed:
                v = plan[k_] if isinstance(plan[k_], list) else [plan[k_]]
                v = [x for x in v if x in allowed]
                plan[k_] = v if len(v) != 1 else v[0]
                if not v: plan.pop(k_)
        return plan
    j = j if isinstance(j, dict) else {}
    plans = [p for p in (clean(p) for p in (j.get("plans") or [])[:5]) if p]
    if len(plans) >= 2:
        return dict(plan=None, plans=plans, conclude=str(j.get("conclude") or "")[:300], modelCalls=1, tokens=tok)
    return dict(plan=clean(j.get("plan")), modelCalls=1, tokens=tok)


CONCLUDE_PROMPT = """A user asked a question about construction drawings. Code measured the facts below on the drawings; they are the
only evidence. Answer the question in 2-4 plain sentences. Use only these facts and cite each one you use as [n]. Never add a
number, size or rule that is not in the facts, except a standard you name as a standard (e.g. "the usual 1.5 m turning circle").
If the facts are not enough to decide, say what is missing. Do not hedge beyond that.

Question: {q}
To decide: {c}
Facts (JSON): {f}
Return JSON only: {{"text": "..."}}"""


def conclude(q, c, facts):
    """Write a short answer to a judgement question from facts code measured; the model reasons, it does not measure."""
    from google import genai
    from google.genai import types
    r = genai.Client(api_key=os.environ["GEMINI_API_KEY"]).models.generate_content(
        model=os.environ.get("CONCLUDE_MODEL") or os.environ.get("PERCEIVE_MODEL", "gemini-4-argon"),
        contents=[CONCLUDE_PROMPT.format(q=q, c=c, f=json.dumps(facts, ensure_ascii=False)[:8000])],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    u = getattr(r, "usage_metadata", None)
    tok = sum((getattr(u, k, 0) or 0) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")) if u else 0
    try: text = str(json.loads(r.text).get("text") or "")
    except Exception: text = r.text
    return dict(text=text[:1500], modelCalls=1, tokens=tok)


VERIFY_PROMPT = """You are checking another reader's work on an architectural floor plan.
Image 1 is the drawing. Image 2 is the same drawing with what the reader found drawn on it: blue outlines with an id
(R1, R2 ...) are rooms, coloured boxes with an id (I1, I2 ...) are objects. The reader's list is below.
Coordinates: a 0-1000 grid over the WHOLE image (x from left, y from top).
Report only clear mistakes you can see; do not re-measure anything. Return JSON only:
{"wrong_items": [{"id": "I7", "actual": "window"|"door"|"toilet"|"sink"|"kitchen_sink"|"shower"|"bathtub"|"stove"|null (null = nothing there)}],
 "missing_items": [{"kind": "door"|"window"|"toilet"|"sink"|"kitchen_sink"|"shower"|"bathtub"|"stove", "point": [x, y]}],
 "missing_rooms": [{"name": "as printed", "number": "as printed or null", "point": [x, y] inside the room}],
 "wrong_rooms": [{"id": "R3", "problem": "short reason"}]}
An item counts as missing only if no box covers it. Use empty lists when everything is right.
Reader's list:
"""


def _overlay(img, rooms, items, side=3072):
    s = min(1.0, side / max(img.shape[:2]))
    v = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    lw = max(2, int(max(v.shape[:2]) / 1500)); fs = max(0.45, max(v.shape[:2]) / 4000)
    col = {"door": (0, 140, 255), "window": (255, 160, 0), "toilet": (0, 170, 0), "sink": (200, 0, 200), "kitchen_sink": (120, 0, 200),
           "shower": (0, 180, 180), "bathtub": (0, 110, 110), "stove": (0, 0, 200)}
    for r in rooms:
        p = (np.float32(r["polygon"]) * s).astype(np.int32)
        if len(p) < 3: continue
        cv2.polylines(v, [p], True, (255, 80, 0), lw)
        x, y = p.mean(0).astype(int); cv2.putText(v, r["id"], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (255, 40, 0), lw)
    for it in items:
        b = [int(t * s) for t in it["box"]]; c = col.get(it["kind"], (90, 90, 90))
        cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), c, lw)
        cv2.putText(v, it["id"], (b[0], max(10, b[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, fs, c, max(1, lw - 1))
    return v, s


def verify_read(img, sch, rooms, items, ppm, perc, model=None):
    """One more model call: shown the drawing and our overlay, the model names mistakes. Each claim is checked in code
    before anything changes; measurements are never taken from the model. Returns (rooms, items, log, tokens)."""
    from google import genai
    from google.genai import types
    H, W = img.shape[:2]
    model = model or os.environ.get("VERIFY_MODEL") or os.environ.get("PERCEIVE_MODEL", "gemini-4-argon")
    for i, it in enumerate(items, 1): it["id"] = f"I{i}"
    ov, s = _overlay(img, rooms, items)
    s0 = min(1.0, 3072 / max(H, W))
    base = cv2.resize(img, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA) if s0 < 1 else img
    listing = json.dumps(dict(rooms=[dict(id=r["id"], label=r.get("label"), areaM2=r.get("measuredAreaM2")) for r in rooms],
                              items=[dict(id=i["id"], kind=i["kind"]) for i in items]), ensure_ascii=False)
    jpg = lambda a: types.Part.from_bytes(data=cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), mime_type="image/jpeg")
    r = genai.Client(api_key=os.environ["GEMINI_API_KEY"]).models.generate_content(
        model=model, contents=[jpg(base), jpg(ov), VERIFY_PROMPT + listing],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    v = json.loads(r.text); u = getattr(r, "usage_metadata", None)
    tok = sum((getattr(u, k, 0) or 0) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")) if u else 0
    log = []
    byid = {i["id"]: i for i in items}
    to_px = lambda p: (float(p[0]) * W / 1000, float(p[1]) * H / 1000)
    hit = lambda b, c: b[0] < c[2] and c[0] < b[2] and b[1] < c[3] and c[1] < b[3]
    door_arc = lambda box: (arcs.best_arc(img, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), ppm, search_m=0.5) or (0,))[0] >= 0.6
    drop = set()
    for w in v.get("wrong_items") or []:
        it = byid.get(str(w.get("id")))
        if not it: continue
        actual = w.get("actual")
        if it["kind"] == "door" and door_arc(it["box"]):
            log.append(f"kept {it['id']} door: the model doubted it, but a door swing arc is drawn there"); continue
        if actual is None and it.get("source") == "detector" and it.get("conf", 0) >= 0.8:
            log.append(f"kept {it['id']} {it['kind']}: confident detection, the model's doubt alone is not enough"); continue
        if actual == "door" and not door_arc(it["box"]):
            log.append(f"kept {it['id']} {it['kind']}: the model said door, but no door arc is drawn there"); continue
        if actual:
            log.append(f"{it['id']}: {it['kind']} -> {actual}"); it["kind"] = actual; it["checked"] = "relabelled by the check"
        else:
            log.append(f"removed {it['id']} {it['kind']}: nothing there"); drop.add(it["id"])
    items = [i for i in items if i["id"] not in drop]
    r_ = max(8, int(0.008 * max(H, W)))
    for m in v.get("missing_items") or []:
        try: px_, py_ = to_px(m["point"])
        except Exception: continue
        kind = str(m.get("kind"))
        box = [int(px_ - r_), int(py_ - r_), int(px_ + r_), int(py_ + r_)]
        same = [i["box"] for i in items if i["kind"] == kind]
        if kind == "door":                                  # a missing door needs its own, uncounted swing arc
            a = arcs.best_arc(img, (px_, py_), ppm, search_m=1.0, exclude=[[b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6] for b in same])
            if not (a and a[0] >= 0.6 and not any(hit(a[1], b) for b in same)):
                log.append(f"ignored a missing door at ({int(px_)},{int(py_)}): no uncounted door arc there"); continue
            box = [int(t) for t in a[1]]
        elif any(hit([px_ - 2 * r_, py_ - 2 * r_, px_ + 2 * r_, py_ + 2 * r_], b) for b in same):
            log.append(f"ignored a missing {kind}: one is already marked there"); continue
        in_rooms = [x["id"] for x in rooms if len(x["polygon"]) >= 3 and cv2.pointPolygonTest(np.int32(x["polygon"]), ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), False) >= 0]
        items.append(dict(id=f"I{len(items) + 1}", kind=kind, box=box, roomIds=in_rooms, conf=0.6, source="check"))
        log.append(f"added a {kind} the check found")
    miss = [m for m in (v.get("missing_rooms") or []) if m.get("point")]
    if miss:                                                # grow the missing rooms from the walls; keep only uncovered ones
        try:
            grown = rooms_from_interior(img, sch, dict(rooms=list((perc or {}).get("rooms") or []) + miss), ppm)
            cover = np.zeros((H, W), np.uint8)
            for x in rooms:
                if len(x["polygon"]) >= 3: cv2.fillPoly(cover, [np.int32(x["polygon"])], 1)
            for g_ in grown:
                if len(g_["polygon"]) < 3: continue
                mm = np.zeros((H, W), np.uint8); cv2.fillPoly(mm, [np.int32(g_["polygon"])], 1); a_ = int(mm.sum())
                if a_ == 0 or (mm & cover).sum() > 0.3 * a_: continue
                cv2.fillPoly(cover, [np.int32(g_["polygon"])], 1)
                rooms.append(dict(g_, id=f"R{len(rooms) + 1}", source="check")); log.append(f"added room {g_.get('label') or ''} the check found".strip())
        except Exception:
            traceback.print_exc()
    for w in v.get("wrong_rooms") or []:                    # rooms are measured by code: a doubt is shown, not applied
        x = next((x for x in rooms if x["id"] == w.get("id")), None)
        if x: x["checkNote"] = str(w.get("problem"))[:160]; log.append(f"flagged {x['id']}: {x['checkNote']}")
    for i, it in enumerate(items, 1): it["id"] = f"I{i}"
    return rooms, items, log, tok


def on_walls(img, rooms, ppm):
    """Share of the room outlines that lies on drawn ink (walls, partitions). Outlines that wander through open floor
    are wrong whatever drew them."""
    if not rooms: return 0.0
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = int(0.12 * (ppm or 59)) | 1
    ink = cv2.dilate((g < 170).astype(np.uint8), np.ones((k, k), np.uint8))
    m = np.zeros((H, W), np.uint8)
    for r in rooms:
        if len(r["polygon"]) >= 3: cv2.polylines(m, [np.int32(r["polygon"])], True, 1, 1)
    return float((m & ink).sum()) / max(1, int(m.sum()))


def to_blueline(sch, img, sheet, file_name, ms):
    H, W = img.shape[:2]
    ppm = sch["scale"]["px_per_m"] if sch["scale"] else None
    rooms = []
    for i, r in enumerate(sch["rooms"], 1):
        printed = None
        for l in r["labels"]:                               # an area printed inside the room, e.g. '25,3 m²'
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*m(?:2|²)", l)
            if m: printed = float(m.group(1).replace(",", "."))
        name = [l for l in r["labels"] if not re.search(r"m(?:2|²)", l)]
        lb = r["label_boxes"]
        rooms.append(dict(id=f"R{i}", type=room_type(name), label=" ".join(name), polygon=[[int(x), int(y)] for x, y in r["polygon"]],
                          printedAreaM2=printed, measuredAreaM2=r["area_m2"], conf=0.9,
                          labelBox=[min(b[0] for b in lb), min(b[1] for b in lb), max(b[2] for b in lb), max(b[3] for b in lb)] if lb else None))
    items = []
    fp = None
    try:
        fp = ask.floorplan(img, ppm=ppm)                    # trained room segmenter + symbol detector, run at its training scale
    except Exception:
        traceback.print_exc(); fp = None
    perc = None
    painted = sch.get("rooms_source") == "painted"
    if True:                                                # every sheet: the model says which rooms and objects there are
        try:
            perc = perceive(img)
        except Exception as e:
            print(f"PERCEPTION FAILED ({type(e).__name__}: {str(e)[:120]}) -> falling back to the model-free reader", flush=True); perc = None
        new_scale = False
        sbm = (perc or {}).get("scale_bar") or {}
        try: mb = [float(v) for v in sbm.get("box")]; mb = [mb[0] * W / 1000, mb[1] * H / 1000, mb[2] * W / 1000, mb[3] * H / 1000]
        except Exception: mb = None
        ob = (sch.get("scale") or {}).get("box")
        pad_ = 0.03 * max(H, W)
        elsewhere = bool(mb and ob) and not (ob[0] < mb[2] + pad_ and mb[0] - pad_ < ob[2] and ob[1] < mb[3] + pad_ and mb[1] - pad_ < ob[3])
        old_ppm = ppm
        if elsewhere:                                       # the model sees the scale bar somewhere else than the line we measured
            print("scale: our bar and the model's scale bar disagree; measuring the model's bar", flush=True)
            ppm = None
        if perc and not ppm and perc.get("scale_bar"):     # no scale found by OCR: measure the bar the model points at
            try:
                p_, box_ = ask.scale_ppu(img, perc["scale_bar"])
                if p_ and 4 <= max(H, W) / p_ <= 3000:
                    ppm = p_; new_scale = True
                    if old_ppm:                             # painted rooms were measured with the old scale
                        for r in rooms:
                            if r.get("measuredAreaM2"): r["measuredAreaM2"] = round(r["measuredAreaM2"] * (old_ppm / ppm) ** 2, 2)
                    sch["scale"] = dict(px_per_m=p_, label=f"{perc['scale_bar'].get('length')} {perc['scale_bar'].get('unit', 'm')} (bar located by the model, measured in pixels)", box=box_, bar_px=None)
            except Exception:
                traceback.print_exc()
        if not ppm: ppm = old_ppm                          # the model's bar could not be measured: keep ours
        if new_scale:                                       # rerun the models at their training scale, now that it is known
            try: fp = ask.floorplan(img, ppm=ppm)
            except Exception: traceback.print_exc()
    cad_like = bool(ppm) and max(H, W) >= 2500             # the trained segmenter's domain (large CAD sheets with a scale)
    # not an architectural plan (piping, electrical ...): lines are pipes and cables, not walls, so no rooms are grown
    # and the door/window detector is not used; the sheet's equipment comes from the symbols list, checked against OCR tags
    plan_sheet = str((perc or {}).get("sheet_type") or "floor_plan") == "floor_plan"
    if not plan_sheet:
        rooms = []
    if not plan_sheet:
        pass
    elif painted and perc and perc.get("rooms"):
        # painted rooms are measured exactly from their fill, but rooms left unpainted (or painted like the background)
        # would be missing: rooms the model lists that no painted room covers are added from the drawing's walls
        try:
            extra = rooms_from_perception(img, sch, fp, perc, ppm) if (fp and cad_like) else rooms_from_interior(img, sch, perc, ppm)
        except Exception:
            traceback.print_exc(); extra = []
        cover = np.zeros((H, W), np.uint8)
        for r in rooms:
            if len(r["polygon"]) >= 3: cv2.fillPoly(cover, [np.int32(r["polygon"])], 1)
        for r in extra:
            if len(r["polygon"]) < 3: continue
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.int32(r["polygon"])], 1)
            a_ = int(m.sum())
            if a_ == 0 or (m & cover).sum() > 0.3 * a_: continue      # already a painted room
            cv2.fillPoly(cover, [np.int32(r["polygon"])], 1)
            rooms.append(dict(r, id=f"R{len(rooms) + 1}", source="model"))
    elif perc and perc.get("rooms") and fp and cad_like:
        # two ways to outline the listed rooms: our trained segmenter (strong on the drawing style it was trained on) and
        # growing each room from its point out to the drawn walls. Keep the one whose outlines actually lie on the walls.
        seg = rooms_from_perception(img, sch, fp, perc, ppm)
        try: grown = rooms_from_interior(img, sch, perc, ppm)
        except Exception: traceback.print_exc(); grown = []
        a_, b_ = on_walls(img, seg, ppm), on_walls(img, grown, ppm)
        rooms = seg if (a_ >= b_ - 0.03 and len(seg) >= 0.8 * len(grown)) else grown
        room_method = f"segmenter ({a_:.0%} of outline on walls) vs grown from walls ({b_:.0%}): kept {'segmenter' if rooms is seg else 'grown from walls'}"
        print("rooms:", room_method, flush=True)
    elif perc and perc.get("rooms"):                       # other styles: rooms straight from the drawing's walls
        rooms = rooms_from_interior(img, sch, perc, ppm)
    elif sch.get("rooms_source") != "painted" and fp:      # no perception available: label-based filtering of the segmenter
        rooms = []
        ROOMNO = re.compile(r"(?<![\d.])\d{1,3}\.\d{1,3}(?![\d.])")
        AREA = re.compile(r"(\d+(?:[.,]\d+)?)\s*m(?:2|²)")
        ctr = lambda t: ((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
        # does this sheet label its rooms at all? If not, a region needs no label to count as a room
        labelled_sheet = any(ROOMNO.search(t["text"]) or room_type([t["text"]]) != "Room" for t in sch["texts"])
        for seg in [r for r in fp["rooms"] if len(r["polygon"]) >= 3]:
            poly = np.int32(seg["polygon"])
            inside = [t for t in sch["texts"] if cv2.pointPolygonTest(poly, tuple(float(v) for v in ctr(t)), False) >= 0]
            nums = [t for t in inside if ROOMNO.search(t["text"]) and not AREA.search(t["text"])]
            if labelled_sheet and not nums and room_type([t["text"] for t in inside]) == "Room":
                continue                                    # title-block cells, legend boxes: no room number, no room name
            if not labelled_sheet and ppm and cv2.contourArea(poly) < 1.5 * ppm ** 2:
                continue                                    # unlabelled sheet: keep regions of room size only
            # one region holding several room numbers = rooms the model merged: split it, each pixel to the nearest number
            x0, y0, w, h = cv2.boundingRect(poly)
            m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [poly - [x0, y0]], 1)
            parts = []
            if len(nums) >= 2:
                yy, xx = np.mgrid[0:h, 0:w]
                d = np.stack([(xx - (ctr(t)[0] - x0)) ** 2 + (yy - (ctr(t)[1] - y0)) ** 2 for t in nums])
                owner = d.argmin(0)
                for k, t in enumerate(nums):
                    pm = ((owner == k) & (m > 0)).astype(np.uint8)
                    cs, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cs: parts.append((t, max(cs, key=cv2.contourArea).reshape(-1, 2) + [x0, y0], float(pm.sum())))
            else:
                parts.append((nums[0] if nums else None, poly.reshape(-1, 2), float(m.sum())))
            for t_no, pp, area_px in parts:
                pp = np.int32(pp)
                txt = [t for t in inside if cv2.pointPolygonTest(pp, tuple(float(v) for v in ctr(t)), False) >= 0]
                txt.sort(key=lambda t: (t["box"][1], t["box"][0]))
                measured = round(area_px / ppm ** 2, 2) if ppm else None
                printed = None                              # the area printed nearest this room's number (usually just below it)
                ar = [t for t in txt if AREA.search(t["text"])]
                if ar:
                    ref = ctr(t_no) if t_no else ctr(txt[0])
                    near = min(ar, key=lambda t: (ctr(t)[0] - ref[0]) ** 2 + (ctr(t)[1] - ref[1]) ** 2)
                    printed = float(AREA.search(near["text"]).group(1).replace(",", "."))
                    if measured and abs(printed / measured - 1) > 0.25:
                        printed = None                      # contradicts the drawn room: an OCR misread, trust the geometry
                name = [t["text"] for t in txt if not AREA.search(t["text"])][:4]
                rooms.append(dict(id=f"R{len(rooms) + 1}", type=room_type(name), label=" ".join(name), polygon=pp.tolist(),
                                  printedAreaM2=printed, measuredAreaM2=printed if printed else measured, conf=0.8,
                                  labelBox=[min(t["box"][0] for t in txt), min(t["box"][1] for t in txt), max(t["box"][2] for t in txt), max(t["box"][3] for t in txt)] if txt else None))
    # Which source counts the objects: our Swiss-trained detector only on large CAD sheets with a measured scale (its
    # domain: 89% exact counts there); elsewhere (scans, CubiCasa-style flats) it is worse than the model, so the model's
    # own item list is used, every item with its point so it is marked on the drawing.
    if perc and perc.get("items") and not cad_like:
        r_ = max(8, int(0.012 * max(H, W)))
        for it in perc["items"]:
            try: px_, py_ = float(it["point"][0]) * W / 1000, float(it["point"][1]) * H / 1000
            except Exception: continue
            kind = {"washbasin": "sink", "stove": "stove"}.get(str(it.get("kind")), str(it.get("kind")))
            in_rooms = [r["id"] for r in rooms if len(r["polygon"]) >= 3 and cv2.pointPolygonTest(np.int32(r["polygon"]), (px_, py_), False) >= 0]
            items.append(dict(id=f"I{len(items) + 1}", kind=kind, box=[int(px_ - r_), int(py_ - r_), int(px_ + r_), int(py_ + r_)],
                              roomIds=in_rooms, conf=0.6, source="model"))
    if plan_sheet and fp and sch.get("rooms_source") != "painted" and (cad_like or not (perc and perc.get("items"))):
        kinds = {"washbasin": "sink", "sink": "kitchen_sink"}            # "stove" is a kitchen stove, not a sauna stove
        for j, it in enumerate([i for i in fp.get("items", []) if i.get("conf", 1) >= 0.3], 1):
            b = it.get("box") or [it["point"][0] - 10, it["point"][1] - 10, it["point"][0] + 10, it["point"][1] + 10]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            in_rooms = [r["id"] for r in rooms if cv2.pointPolygonTest(np.int32(r["polygon"]), (float(cx), float(cy)), False) >= 0]
            items.append(dict(id=f"I{j}", kind=kinds.get(it["type"], it["type"]), box=[int(v) for v in b], roomIds=in_rooms, conf=round(it.get("conf", 1), 3)))
    # count reconciliation: per object type, when the model lists MORE than the detector found (0, 1 or 2 doors
    # where the model saw 12), the model's points that no detected box explains are added (source "model") until
    # the count matches the model's. Where the detector finds as many or more, its boxes stand.
    if perc and perc.get("items"):
        R = (1.2 * ppm) if ppm else 0.03 * max(H, W)          # a model point within ~1.2 m of a detected box is that box
        r_ = max(8, int(0.008 * max(H, W)))
        mod = {}
        for it in perc["items"]:
            try: p = (float(it["point"][0]) * W / 1000, float(it["point"][1]) * H / 1000)
            except Exception: continue
            mod.setdefault({"washbasin": "sink"}.get(str(it.get("kind")), str(it.get("kind"))), []).append(p)
        for kind, pts in mod.items():
            det = [((i["box"][0] + i["box"][2]) / 2, (i["box"][1] + i["box"][3]) / 2) for i in items if i["kind"] == kind]
            gap = len(pts) - len(det)
            if gap <= 0: continue
            free = list(det)
            unexplained = []
            for p in pts:                                     # greedy one-to-one match, each box explains one point
                if free:
                    d = [np.hypot(p[0] - q[0], p[1] - q[1]) for q in free]
                    j = int(np.argmin(d))
                    if d[j] <= R: free.pop(j); continue
                    unexplained.append((d[j], p))
                else:
                    unexplained.append((1e9, p))
            unexplained.sort(key=lambda t: -t[0])              # farthest from any detection first
            taken = [i["box"] for i in items if i["kind"] == "door"]
            for _, (px_, py_) in unexplained[:gap]:
                box = [int(px_ - r_), int(py_ - r_), int(px_ + r_), int(py_ + r_)]
                if kind == "door":
                    # the model's points are rough: a door goes on the swing arc next to its point. When the detector
                    # did find doors, a model door with no uncounted arc nearby is not added (it would be a window,
                    # or a door the model counted twice)
                    try: a = arcs.best_arc(img, (px_, py_), ppm, search_m=1.0, exclude=[[b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6] for b in taken])
                    except Exception: a = None
                    hit = lambda b, c: b[0] < c[2] and c[0] < b[2] and b[1] < c[3] and c[1] < b[3]
                    if a and a[0] >= 0.6 and not any(hit(a[1], t) for t in taken): box = [int(v) for v in a[1]]
                    elif det: continue                      # no uncounted arc: an already counted door, or not a door
                    taken.append(box)
                cx_, cy_ = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                in_rooms = [r["id"] for r in rooms if len(r["polygon"]) >= 3 and cv2.pointPolygonTest(np.int32(r["polygon"]), (cx_, cy_), False) >= 0]
                items.append(dict(id=f"I{len(items) + 1}", kind=kind, box=box, roomIds=in_rooms, conf=0.6, source="model"))
    # guard: a room the model listed whose outline we could not find is still reported (counts, names, questions),
    # marked outlineFound=False with no area, instead of silently missing
    if plan_sheet and perc and perc.get("rooms"):
        r0 = max(6, int(0.006 * max(H, W)))
        for pr in perc["rooms"]:
            try: px_, py_ = float(pr["point"][0]) * W / 1000, float(pr["point"][1]) * H / 1000
            except Exception: continue
            if any(len(r["polygon"]) >= 3 and cv2.pointPolygonTest(np.int32(r["polygon"]), (px_, py_), True) > -0.01 * max(H, W) for r in rooms):
                continue
            label = " ".join(x for x in (str(pr.get("name") or "").strip(), str(pr.get("number") or "").strip()) if x)
            if label and any((r.get("label") or "").strip().lower() == label.lower() for r in rooms): continue
            rooms.append(dict(id=f"R{len(rooms) + 1}", type=room_type([label]), label=label,
                              polygon=[[int(px_ - r0), int(py_ - r0)], [int(px_ + r0), int(py_ - r0)], [int(px_ + r0), int(py_ + r0)], [int(px_ - r0), int(py_ + r0)]],
                              printedAreaM2=None, measuredAreaM2=None, conf=0.3, source="model", outlineFound=False))
    # printed areas the model read (OCR misses rotated, upside-down or tiny text) go on the room that holds them
    if plan_sheet and perc and perc.get("rooms"):
        for pr in perc["rooms"]:
            try: a = float(str(pr.get("area")).replace(",", ".")) if pr.get("area") not in (None, "", "null") else None
            except Exception: a = None
            if not a or not (0.3 <= a <= 5000): continue
            try: px_, py_ = float(pr["point"][0]) * W / 1000, float(pr["point"][1]) * H / 1000
            except Exception: continue
            host = [r for r in rooms if len(r["polygon"]) >= 3 and cv2.pointPolygonTest(np.int32(r["polygon"]), (px_, py_), False) >= 0]
            if len(host) == 1 and host[0].get("printedAreaM2") is None:
                host[0]["printedAreaM2"] = round(a, 2); host[0]["printedAreaSource"] = "read by the model"
    # no scale bar found: the printed room areas give the scale (pixel area / printed area, median over the rooms,
    # used only when most rooms agree), and every room is then measured with it
    if plan_sheet and not ppm:
        pa = [(r, cv2.contourArea(np.float32(r["polygon"])) / r["printedAreaM2"]) for r in rooms
              if r.get("printedAreaM2") and len(r["polygon"]) >= 3 and r.get("outlineFound", True)]
        pa = [(r, v) for r, v in pa if v > 0]
        if len(pa) >= 3:
            med = float(np.median([v for _, v in pa]))
            agree = sum(abs(v / med - 1) <= 0.15 for _, v in pa) / len(pa)
            if agree >= 0.6:
                ppm = med ** 0.5
                sch["scale"] = dict(px_per_m=ppm, label=f"from {len(pa)} printed room areas ({agree:.0%} agree within 15%)", box=None, bar_px=None)
                for r in rooms:
                    if len(r["polygon"]) >= 3 and r.get("outlineFound", True):
                        r["measuredAreaM2"] = round(cv2.contourArea(np.float32(r["polygon"])) / ppm ** 2, 2)
                print("scale:", sch["scale"]["label"], flush=True)
    if perc and perc.get("symbols"):
        norm = lambda t: re.sub(r"[^A-Z0-9]", "", str(t).upper())
        ocr = {}
        for t in sch["texts"]:
            for m in re.finditer(r"(?<![A-Z0-9])([A-Z]{1,6})\s?-\s?(\d{1,4}[A-Z]?)(?![\d.])", t["text"].upper()):
                ocr.setdefault(m.group(1) + m.group(2), []).append(t["box"])
        used, prefix_kind = set(), {}
        rs_ = max(8, int(0.008 * max(H, W)))
        for sy in perc["symbols"]:
            try: px_, py_ = float(sy["point"][0]) * W / 1000, float(sy["point"][1]) * H / 1000
            except Exception: continue
            kind = re.sub(r"\s+", " ", str(sy.get("kind") or "symbol").strip().lower())[:40]
            tag = norm(sy.get("tag") or "")
            box, src = [int(px_ - rs_), int(py_ - rs_), int(px_ + rs_), int(py_ + rs_)], "model"
            if tag and tag in ocr:                          # the printed tag is really there: put the box on it
                b_ = min(ocr[tag], key=lambda b: (b[0] + b[2] - 2 * px_) ** 2 + (b[1] + b[3] - 2 * py_) ** 2)
                box, src = [int(v) for v in b_], "tag read by OCR"; used.add(tag)
                prefix_kind.setdefault(re.match(r"[A-Z]+", tag).group(0), kind)
            items.append(dict(id=f"I{len(items) + 1}", kind=kind, tag=sy.get("tag"), box=box, roomIds=[], conf=0.8 if src != "model" else 0.6, source=src))
        for tag, boxes in ocr.items():                      # tagged objects the model did not list, named like their siblings
            p_ = re.match(r"[A-Z]+", tag).group(0)
            if tag in used or p_ not in prefix_kind: continue
            items.append(dict(id=f"I{len(items) + 1}", kind=prefix_kind[p_], tag=tag, box=[int(v) for v in boxes[0]], roomIds=[], conf=0.7, source="tag read by OCR"))
    for it in items: it.setdefault("source", "detector")
    check_log, check_tok = None, 0
    if perc and os.environ.get("VERIFY", "1") != "0":       # second look (VERIFY=0 turns it off): the model reviews our read, code checks each claim
        try:
            rooms, items, check_log, check_tok = verify_read(img, sch, rooms, items, ppm, perc)
        except Exception as e:
            print(f"CHECK FAILED ({type(e).__name__}: {str(e)[:160]}) -> keeping the first read", flush=True); check_log = None
        if not plan_sheet: rooms = []                       # the check cannot add rooms to a non-architectural sheet
    clean = []                                              # every room leaves with a valid outline (>= 3 integer points)
    for r in rooms:
        pts_ = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in (r.get("polygon") or []) if len(p) >= 2]
        if len(pts_) < 3:
            if not pts_: continue
            cx_, cy_ = np.mean(pts_, 0); r0 = max(6, int(0.006 * max(H, W)))
            pts_ = [[int(cx_ - r0), int(cy_ - r0)], [int(cx_ + r0), int(cy_ - r0)], [int(cx_ + r0), int(cy_ + r0)], [int(cx_ - r0), int(cy_ + r0)]]
            r["outlineFound"] = False; r["measuredAreaM2"] = None
        r["polygon"] = pts_; clean.append(r)
    remap = {}
    for i, r in enumerate(clean, 1): remap[r["id"]] = f"R{i}"; r["id"] = f"R{i}"
    rooms = clean
    for it in items: it["roomIds"] = [remap[x] for x in (it.get("roomIds") or []) if x in remap]
    tags = [dict(prefix=p, count=d["count"], numbers=d["numbers"], boxes=d["boxes"]) for p, d in sch["tags"].items()]
    sc = sch["scale"]
    return dict(
        drawing=dict(id=str(sheet.get("id") or os.path.splitext(file_name)[0]), file=file_name, width=W, height=H,
                     metersPerPixel=(1 / ppm) if ppm else None,
                     scale=dict(method="scale bar measured in pixels + its printed numbers (OCR)", text=sc["label"], box=sc["box"]) if sc else dict(method="none")),
        rooms=rooms, items=items, walls=[], texts=sch["texts"], tags=tags,
        meta=dict(source="blueline", version="perceive+measure v2", msRead=ms, costEUR=0.0, modelCalls=(1 if perc else 0) + (1 if check_log is not None else 0), tokens=(perc or {}).get("_tokens", 0) + check_tok, check=check_log,
                  note=("one model call says which rooms there are; " if perc else "no model call; ") + "measured: scale bar, OCR-checked labels, "
                       + ("painted-room geometry" if sch.get("rooms_source") == "painted" else "trained room segmenter") + ("; trained detector for doors/windows/fixtures" if fp else "")))


def parse_multipart(handler):
    n = int(handler.headers.get("Content-Length", 0)); body = handler.rfile.read(n)
    msg = BytesParser(policy=email_default).parsebytes(b"Content-Type: " + handler.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
    out = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        out[name] = (part.get_filename(), part.get_payload(decode=True))
    return out


def load_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:                                         # a PDF: render its first page at 300 dpi
        import fitz
        doc = fitz.open(stream=data, filetype="pdf"); pix = doc[0].get_pixmap(dpi=300)
        img = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3], cv2.COLOR_RGB2BGR)
    return img


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in (("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"), ("Access-Control-Allow-Headers", "*")):
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")
        if p.endswith("/health"): return self._send(200, {"ok": True, "reader": "blueline-deterministic"})
        if p.endswith("/analyze") or p.endswith("/ask") or p.endswith("/plan") or p.endswith("/conclude"):
            return self._send(405, {"error": "use POST"})          # the page's probe: 405 here means "our reader, use /analyze"
        if p.endswith("/api/scene"): return self._send(404, {"error": "plan not found"})   # lets the page detect us and use /analyze
        if p in ("", "/index.html"):                            # serve the frontend itself: one URL for page + reader
            page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
            if os.path.exists(page):
                b = open(page, "rb").read()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); return self.wfile.write(b)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0].rstrip("/")
        if p.endswith("/plan") and not p.endswith("/api/plan"):   # question -> plan the page runs in code (PLAN_API.md)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
                return self._send(200, plan_question(str(body.get("question", ""))[:500], body.get("vocabulary") or {}))
            except Exception as e:
                traceback.print_exc(); return self._send(200, {"plan": None, "modelCalls": 1, "tokens": 0, "error": repr(e)[:200]})
        if p.endswith("/conclude"):                                 # several measured answers -> a short reasoned answer
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
                return self._send(200, conclude(str(body.get("question", ""))[:500], str(body.get("conclude", ""))[:300], body.get("facts") or []))
            except Exception as e:
                traceback.print_exc(); return self._send(200, {"text": "", "modelCalls": 1, "tokens": 0, "error": repr(e)[:200]})
        if not (p.endswith("/analyze") or p.endswith("/ask")):      # e.g. /api/plan: answer 404 at once, the page falls back to /analyze
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)); return self._send(404, {"error": "not found"})
        try:
            f = parse_multipart(self)
            fname, data = f.get("file", (None, None))
            if not data: return self._send(400, {"error": "missing file"})
            if self.path.rstrip("/").endswith("/analyze"):
                # result cache keyed by the file's exact content: reading the same file again returns the earlier read
                import hashlib
                digest = hashlib.sha256(data + READER_VERSION.encode() + os.environ.get("PERCEIVE_MODEL", "gemini-4-argon").encode()).hexdigest()
                cpath = os.path.join(CACHE_DIR, digest + ".json")
                sheet = json.loads((f.get("sheet") or (None, b"{}"))[1] or b"{}")
                if os.path.exists(cpath):
                    out = json.load(open(cpath))
                    out["drawing"]["id"] = str(sheet.get("id") or out["drawing"]["id"]); out["drawing"]["file"] = fname or out["drawing"]["file"]
                    out["meta"]["cached"] = True
                    out["meta"]["note"] = f"cached: this exact file was read before ({out['meta'].get('readAt', '?')}); " + out["meta"].get("note", "")
                    return self._send(200, out)
                img = load_image(data); t0 = time.time()
                sch = schematic.build(img, key=ask._img_key(img))
                out = to_blueline(sch, img, sheet, fname or "sheet.png", int((time.time() - t0) * 1000))
                out["meta"]["readAt"] = time.strftime("%Y-%m-%d %H:%M:%S"); out["meta"]["cached"] = False
                try:
                    os.makedirs(CACHE_DIR, exist_ok=True); json.dump(out, open(cpath, "w"), ensure_ascii=False)
                except Exception:
                    pass
                return self._send(200, out)
            img = load_image(data); t0 = time.time()
            sch = schematic.build(img, key=ask._img_key(img))
            if self.path.rstrip("/").endswith("/ask"):
                q = (f.get("question") or (None, b""))[1].decode()
                r = schematic.answer(sch, q)
                if r:
                    v, ev = r
                    evid = {"polygons": [ev["room"]["polygon"]], "boxes": ev["room"]["label_boxes"]} if ev["kind"] == "room" else {"boxes": ev["tags"]["boxes"], "polygons": []}
                    return self._send(200, dict(answer=v, unit="m²" if ev["kind"] == "room" else "", status="confident", source="schematic (no model call)",
                                                modelCalls=0, tokens=0, methods=[], evidence=evid))
                plan = ask.plan_with_gemini(img, q, os.environ.get("ASK_MODEL", "gemini-3.8-flash"))
                res, ev = ask.solve(img, plan); u = plan.get("_usage", {})
                a_ = res["answer"]
                # only the drawing: no free text from the model (a recipe, an essay); a model-only answer must be a short value
                if a_ is None or (res["source"].startswith("model") and (not isinstance(a_, (int, float)) and (len(str(a_)) > 40 or ask.num(a_) is None))):
                    return self._send(200, dict(answer=None, status="cant_verify", source="not answerable from this drawing",
                                                modelCalls=1, tokens=u.get("in", 0) + u.get("out", 0) + u.get("thinking", 0), methods=[], evidence={}))
                polys = []
                if ev.get("mask") is not None:
                    cs, _ = cv2.findContours(np.asarray(ev["mask"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polys = [c.reshape(-1, 2).tolist() for c in cs if cv2.contourArea(c) > 50]
                return self._send(200, dict(answer=res["answer"], unit=plan.get("answer_unit") or "", status=res["status"], source=res["source"],
                                            modelCalls=1, tokens=u.get("in", 0) + u.get("out", 0) + u.get("thinking", 0),
                                            methods=[{"tool": m["tool"], "value": m["value"], "detail": str(m["detail"])[:200]} for m in res["runs"]],
                                            evidence={"boxes": [b["box"] for b in ev.get("boxes", [])], "polygons": polys, "lines": ev.get("lines", [])}))
            self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc(); self._send(500, {"error": repr(e)[:500]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Blueline API on http://localhost:{port}  (POST /analyze, POST /ask, GET /health)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
