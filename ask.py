"""Ask a question about any drawing. A language model plans; deterministic tools compute; the evidence is drawn on the sheet.

    export GEMINI_API_KEY=...
    python3 ask.py --image sheet.png -q "How many pressure gauges are there?"
    python3 ask.py --image sheet.png -q "..." --plan plan.json        # skip the model, run a given plan (testing)

Flow
  1. Gemini sees the sheet + question and returns JSON: its own direct answer, the scale bar (box, length, unit),
     and ONE tool call that computes the answer from the drawing.
  2. The tool runs locally and deterministically (drawing_tools.py, scale_bar.py): OCR tag counts, symbol matching,
     region areas, polyline lengths, or verifying a printed value with OCR.
  3. Confidence: the tool's result is compared with Gemini's direct answer. Agree -> 'confident'. Disagree -> 'check',
     both are reported. The overlay shows exactly what was counted / measured.
"""
import argparse, io, json, math, os, re, sys, time
import numpy as np, cv2
import drawing_tools as T

# key: environment variable, else a .env file next to this script (KEY=value lines; never commit it)
_envf = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_envf):
    for _l in open(_envf):
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.strip().split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

_TEXT_CACHE = {}


def sheet_text(img):
    k = (img.shape, int(img[::97, ::89].sum()))
    if k not in _TEXT_CACHE: _TEXT_CACHE[k] = T.read_text(img)
    return _TEXT_CACHE[k]

PROMPT = """You are planning how to answer a question about a technical drawing (floor plan, piping, structural, ...).
Coordinates: give every point/box in a 0-1000 grid over the WHOLE image: x from left (0) to right (1000), y from top (0) to bottom (1000).
Image size: {W} x {H} px.

Question: {q}

Propose up to THREE independent tool calls that could each compute the answer from the drawing itself (best first).
Independent methods let the answers check each other, e.g. for a count: tags AND symbol matching AND the floor-plan detector;
for an area: a printed value AND printed dimensions AND measuring the region.
For ARCHITECTURAL FLOOR PLANS (rooms, doors, windows, sanitary fixtures) use the trained floor-plan tools first:
- "floorplan_count": doors / windows / toilets / washbasins / bathtubs / showers / stoves / sinks, counted by our trained detector. args: {{"item": "door"}} (one of door, window, toilet, washbasin, bathtub, shower, stove, sink)
- "floorplan_room_area": area of one room, measured from our trained room segmentation and the scale bar. args: {{"room": the room's printed number or name as written, e.g. "1.02" or "Bagno"}}. If the room's area is printed on the plan, use read_value instead.
Other tools:
- "count_tags": objects identified by printed tags like CV-1, FP-3, PS-12. args: {{"prefix": "CV"}}. ALWAYS prefer this when the objects carry numbered tags.
- "count_symbol": repeated drawn symbols (gauges, valves, doors, windows, toilets...). args: {{"example_box": [x0,y0,x1,y1] tightly around ONE clear example, "text": the text inside the symbol if it has any (e.g. "PG") else null}}
- "measure_area": area of one or more regions. args: {{"seeds": [[x,y], ...] one point well inside EACH enclosed part of the region (every panel/room that belongs to it), "unit": "m2" or "ft2"}}
- "measure_length": length along drawn lines (pipes, walls, baseboards). args: {{"points": [[x,y], ...] polyline vertices in order}}
- "from_dimensions": the answer follows from printed dimensions (e.g. a rectangular slab 6200 x 9950 mm). args: {{"values": ["6200", "9950"] exactly as printed, "expression": arithmetic using only those numbers, e.g. "6200*9950/1e6" (convert to the asked unit)}}. Prefer this over measure_area whenever the region's dimensions are printed.
- "read_value": the answer is printed on the sheet (a table, a note, a dimension). args: {{"value": "the exact printed text", "box": [x0,y0,x1,y1] around it}}

Also report the scale bar if there is one (needed for areas/lengths): {{"box": [x0,y0,x1,y1], "length": the largest number printed on the bar, "unit": "m" / "mm" / "ft"}}, else null.

If the question is not about this drawing (or about construction drawings at all), return "direct_answer": null and "calls": [].

Return JSON only:
{{"direct_answer": <number or text: your own best answer>, "answer_unit": "...", "calls": [{{"tool": "...", "args": {{...}}}}, ...], "scale_bar": {{...}} or null, "reasoning": "one sentence"}}"""

UNIT = {"m": 1.0, "mm": 0.001, "cm": 0.01, "ft": 0.3048, "in": 0.0254}


_FP_CACHE = {}
MODELS = {"unet": os.environ.get("FP_UNET", "unet_swiss.pt"), "yolo": os.environ.get("FP_YOLO", "yolo_swiss.pt")}


TRAIN_PPM = 59.0          # px per metre of the Swiss training sheets (1:100 on A3 at 2480 px)


def floorplan(img, ppm=None):
    """Our trained room segmenter + symbol detector on this sheet (CPU is fine; cached per image).
    The models only work at the scale they were trained on, so the sheet is first resized to ~59 px per metre
    (from the scale bar when known, else the long side to the training size), and results are mapped back."""
    H0, W0 = img.shape[:2]
    s = (TRAIN_PPM / ppm) if ppm else (2480 / max(H0, W0) if max(H0, W0) > 3000 else 1.0)
    s = min(4.0, max(0.2, s))
    if abs(s - 1) > 0.08:
        k0 = (img.shape, int(img[::97, ::89].sum()), round(s, 3))
        if k0 in _FP_CACHE: return _FP_CACHE[k0]
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        fp = floorplan(small, ppm=None if not ppm else TRAIN_PPM)
        if fp is None: return None
        out = {"rooms": [dict(r, polygon=[[x / s, y / s] for x, y in r["polygon"]], area_px=r["area_px"] / s / s) for r in fp["rooms"]],
               "items": [dict(i, point=[v / s for v in i["point"]], box=[v / s for v in i["box"]]) for i in fp["items"]], "scaled_by": s}
        _FP_CACHE[k0] = out
        return out
    k = (img.shape, int(img[::97, ::89].sum()))
    if k in _FP_CACHE: return _FP_CACHE[k]
    if not (os.path.exists(MODELS["unet"]) and os.path.exists(MODELS["yolo"])): return None
    import torch, segmentation_models_pytorch as smp
    from ultralytics import YOLO
    from PIL import Image
    from seg_predict import predict_mask, rooms_from_mask, predict_items, nms
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "unet" not in _FP_CACHE:
        u = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1).to(dev)
        u.load_state_dict(torch.load(MODELS["unet"], map_location=dev)); u.eval()
        _FP_CACHE["unet"], _FP_CACHE["yolo"] = u, YOLO(MODELS["yolo"])
    from seg_data import upfactor
    pil = Image.fromarray(img[..., ::-1]); f = upfactor(*pil.size)          # small scans are upscaled like in training
    if f > 1: pil = pil.resize((pil.width * f, pil.height * f), Image.BICUBIC)
    rooms = rooms_from_mask(predict_mask(_FP_CACHE["unet"], np.asarray(pil.convert("L")), dev), min_px=200 * f * f)
    items = predict_items(_FP_CACHE["yolo"], pil)
    items = nms(items, 0.7) if items and "box" in items[0] else items
    if f > 1:
        for r in rooms: r["polygon"] = [[x / f, y / f] for x, y in r["polygon"]]; r["area_px"] = r["area_px"] / f / f
        for i in items: i["point"] = [v / f for v in i["point"]]; i["box"] = [v / f for v in i["box"]]
    _FP_CACHE[k] = {"rooms": rooms, "items": items}
    return _FP_CACHE[k]


def plan_with_gemini(img, q, model):
    from google import genai
    from google.genai import types
    H, W = img.shape[:2]
    s = min(1.0, 3072 / max(H, W))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    ok, buf = cv2.imencode(".png", small)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(model=model, contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/png"),
                                                              PROMPT.format(W=W, H=H, q=q)],
                                       config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    plan = json.loads(r.text)
    u = getattr(r, "usage_metadata", None)              # what this question cost: tokens in / out / thinking
    plan["_question"] = q
    plan["_usage"] = {"model": model, "in": getattr(u, "prompt_token_count", 0) or 0, "out": getattr(u, "candidates_token_count", 0) or 0,
                      "thinking": getattr(u, "thoughts_token_count", 0) or 0}
    return plan


def px(p, W, H):
    return [p[0] * W / 1000, p[1] * H / 1000]


def box_px(b, W, H):
    return [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]


def scale_ppu(img, sb):
    """Pixels per metre from the scale bar: the bar is measured in pixels; its length/unit come from the plan."""
    if not sb or not sb.get("length"): return None, None
    H, W = img.shape[:2]
    metres = float(sb["length"]) * UNIT.get(str(sb.get("unit", "m")).lower(), 1.0)
    from scale_bar import find_bars
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = [v / 1000 for v in sb["box"]] if sb.get("box") else None
    from scale_bar import segmented_in
    # inside the model's box: a bar of separate black blocks first (a long solid rule line can sit right next to it),
    # then an outlined bar, then smaller bars; finally anywhere on the sheet
    bars = (segmented_in(g, box) if box else []) or find_bars(g, box) or (find_bars(g, box, relaxed=True) if box else []) or find_bars(g)
    if not bars: return None, None
    x0, x1, y, L, _ = bars[0]
    return L / metres, [int(x0), int(y) - 12, int(x1), int(y) + 12]


def _snap_text_example(img, b, txt):
    """Move a rough example box onto the symbol around `txt` near it (OCR + drawn circle). None if the text is not near b."""
    W, H = img.shape[1], img.shape[0]
    texts = T.read_text(img[int(max(0, b[1] - 80)):int(b[3] + 80), int(max(0, b[0] - 80)):int(b[2] + 80)])
    near = sorted((t for t in texts if txt.upper() in t["text"].upper().replace(" ", "")),
                  key=lambda t: (len(t["text"].replace(" ", "")) != len(txt), -t["conf"]))   # exact text first
    if near:
        t = near[0]; ox, oy = max(0, b[0] - 80), max(0, b[1] - 80)
        full = t["text"].upper().replace(" ", ""); i = full.find(txt.upper().replace(" ", ""))
        cw = (t["box"][2] - t["box"][0]) / max(1, len(full))      # OCR may glue neighbours on: keep only our text's part
        tx0 = t["box"][0] + i * cw; tw = cw * len(txt.replace(" ", ""))
        cx, cy = tx0 + tw / 2 + ox, (t["box"][1] + t["box"][3]) / 2 + oy
        r = 0.45 * tw                                            # fallback if no circle is found
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); R0 = int(3 * tw) + 20
        y0_, x0_ = int(max(0, cy - R0)), int(max(0, cx - R0))
        c = cv2.HoughCircles(cv2.GaussianBlur(g[y0_:int(cy + R0), x0_:int(cx + R0)], (5, 5), 1), cv2.HOUGH_GRADIENT, 1, 10,
                             param1=100, param2=20, minRadius=int(0.3 * tw), maxRadius=int(0.8 * tw))
        if c is not None:                                       # a circle around the text: use it as the symbol
            cc = min(c[0], key=lambda k: (k[0] + x0_ - cx) ** 2 + (k[1] + y0_ - cy) ** 2)
            if (cc[0] + x0_ - cx) ** 2 + (cc[1] + y0_ - cy) ** 2 < (0.3 * tw) ** 2:
                cx, cy, r = cc[0] + x0_, cc[1] + y0_, cc[2] + 2
        # snap to the drawn circle around the text, if there is one: best ring of dark pixels near the text
        ang = np.linspace(0, 2 * np.pi, 120, endpoint=False); best = (0, cx, cy, r)
        for dy in range(-12, 13, 2):
            for dx in range(-12, 13, 2):
                for rr in range(max(6, int(0.3 * tw)), int(0.9 * tw) + 1):
                    xs = np.clip((cx + dx + rr * np.cos(ang)).astype(int), 0, g.shape[1] - 1)
                    ys = np.clip((cy + dy + rr * np.sin(ang)).astype(int), 0, g.shape[0] - 1)
                    sc = (255 - g[ys, xs].astype(float)).mean() / 255
                    if sc > best[0]: best = (sc, cx + dx, cy + dy, rr)
        if best[0] > 0.25: _, cx, cy, r = best; r += 2
        return [cx - r, cy - r, cx + r, cy + r]
    return None


def _img_key(img):
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return (img.shape, hash(small.tobytes()))


def run_tool(img, plan, texts_cache={}):
    H, W = img.shape[:2]; tool, a = plan.get("tool"), plan.get("args") or {}
    ev = {"boxes": [], "mask": None, "lines": []}
    ppm, bar_box = scale_ppu(img, plan.get("scale_bar"))
    if bar_box: ev["boxes"].append({"box": bar_box, "label": f"scale {ppm:.2f} px/m"})
    if tool == "floorplan_room_area":                      # the sheet's schematic first: painted rooms measured once per sheet
        import schematic
        sch = schematic.build(img, key=_img_key(img))
        r = schematic._room_named(sch, str(a.get("room", "")).strip())
        if r:
            m = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(m, [np.int32(r["polygon"])], 1)
            ev["mask"] = m.astype(bool); ev["boxes"] += [{"box": b, "label": ""} for b in r["label_boxes"]]
            if sch["scale"]: ev["boxes"].append({"box": sch["scale"]["box"], "label": f"scale {sch['scale']['label']}"})
            return r["area_m2"], f"room fill around label '{' '.join(r['labels'])}' (schematic)", ev
    if tool == "floorplan_room_area" and ppm:
        want = str(a.get("room", "")).strip()
        for t in find_label(sheet_text(img), want)[:3]:
            area, reg = room_by_fill(img, t["box"], ppm)
            if area:
                ev["mask"] = reg; ev["boxes"].append({"box": t["box"], "label": want})
                return area, f"room fill around label '{t['text']}'", ev
    if tool in ("floorplan_count", "floorplan_room_area"):
        fp = floorplan(img)
        if fp is None: return None, "floor-plan models not found (set --unet / --yolo)", ev
        if tool == "floorplan_count":
            t = str(a.get("item", "")).lower().rstrip("s").replace("wash basin", "washbasin")
            hits = [i for i in fp["items"] if i["type"] == t and i.get("conf", 1) >= 0.3]
            for h in hits: ev["boxes"].append({"box": h["box"], "label": t})
            return len(hits), f"{len(hits)} {t} detections (trained detector, confidence >= 0.3)", ev
        if not ppm: return None, "no scale found", ev
        want = str(a.get("room", "")).strip()
        from shapely.geometry import Polygon, Point
        segs = [(Polygon(r["polygon"]).buffer(0), r) for r in fp["rooms"] if len(r["polygon"]) >= 3]
        cand = find_label(sheet_text(img), want)
        for t in cand:                                     # the room is the segment that contains its label
            c = Point((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
            inside = [(p, r) for p, r in segs if p.contains(c)]
            if inside:
                p, r = min(inside, key=lambda x: x[0].area)
                m = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(m, [np.int32(r["polygon"])], 1)
                # other room numbers inside the same segment = rooms joined through an open doorway: split it
                others = [o for o in sheet_text(img) if re.fullmatch(r"\d+\.\d+", o["text"].strip()) and o is not t
                          and p.contains(Point((o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2))]
                note = ""
                if others:
                    labs = [t] + others
                    r_ = max(2, int(0.35 * ppm))                   # doorways are ~0.8-1 m wide: erode until the rooms come apart
                    core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1)))
                    n, cc = cv2.connectedComponents(core, connectivity=4)
                    seeds = []
                    for L in labs:                                 # each label claims the core it sits in (or a disc at the label)
                        x, y = int((L["box"][0] + L["box"][2]) / 2), int((L["box"][1] + L["box"][3]) / 2)
                        k = cc[y, x] if 0 <= y < cc.shape[0] and 0 <= x < cc.shape[1] else 0
                        sd = (cc == k).astype(np.uint8) if k and sum(1 for M in labs if cc[int((M["box"][1] + M["box"][3]) / 2), int((M["box"][0] + M["box"][2]) / 2)] == k) == 1 \
                            else cv2.circle(np.zeros_like(m), (x, y), 3, 1, -1)
                        seeds.append(sd)
                    d = np.stack([cv2.distanceTransform((1 - sd).astype(np.uint8), cv2.DIST_L2, 3) for sd in seeds])
                    m = ((d.argmin(0) == 0) & (m > 0)).astype(np.uint8)
                    note = f"; split from rooms {[o['text'].strip() for o in others]} sharing the segment"
                ev["mask"] = m.astype(bool)
                ev["boxes"].append({"box": t["box"], "label": want})
                return float(m.sum()) / ppm ** 2, f"segment containing label '{t['text']}'{note}", ev
        return None, f"label {want!r} not found inside any segmented room", ev
    if tool == "count_tags":
        texts = sheet_text(img)
        a["prefix"] = re.sub(r"[\s\-_#0-9]+$", "", str(a["prefix"]).strip().upper())     # 'FP-' / 'FP-1' -> 'FP'
        a["count"] = "distinct"                                                            # objects, not label occurrences
        r = T.count_tags(texts, a["prefix"])
        nums = r["numbers"]; run = 0
        while run + 1 in nums: run += 1                     # tags are numbered 1..n; an isolated extra number is usually a misread
        extra = [n for n in nums if n > run]
        val = r["distinct"] if a.get("count", "distinct") == "distinct" else r["instances"]
        note = ""
        if run >= 3 and extra and len(extra) <= max(1, run // 4):
            val = run; note = f"; ignored isolated {[f'{a['prefix']}-{n}' for n in extra]} (likely a fragment of another tag)"
        elif nums and len(nums) < max(nums) and len(nums) >= max(nums) / 2:
            val = max(nums); note = f"; tags are numbered 1..{max(nums)}, {sorted(set(range(1, max(nums) + 1)) - set(nums))} not read"
        ev["boxes"] += [{"box": h["box"], "label": h["label"]} for h in r["hits"] if h["num"] not in extra or not note]
        return val, f"{r['instances']} labels, numbers {nums}{note}", ev
    if tool == "count_symbol":
        b = box_px(a["example_box"], W, H); txt = a.get("text")
        if txt:
            # candidates: the model's example first, then every place OCR reads that exact text (the model's box can be off)
            exs = [_snap_text_example(img, b, txt)]
            same = [t for t in sheet_text(img) if t["text"].strip().upper().replace(" ", "") == txt.upper().replace(" ", "")]
            exs += [_snap_text_example(img, t["box"], txt) for t in same[:4]]
            best_r, best_b = None, b
            for e in [e for e in exs if e is not None]:
                r = T.count_symbol(img, e, thresh=0.5, scales=(0.9, 1.0, 1.1), rotations=False, verify_text=txt)
                if best_r is None or r["count"] > best_r["count"]: best_r, best_b = r, e
            if best_r is None: return None, f"text '{txt}' not found near the example or on the sheet", ev
            ev["boxes"] += [{"box": h["box"], "label": txt} for h in best_r["hits"]]
            return best_r["count"], f"example at {[int(v) for v in best_b]} (best of {len([e for e in exs if e is not None])} examples)", ev
        if not txt:                                        # tighten the example to its ink
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); x0, y0, x1, y1 = [int(v) for v in b]
            ys, xs = np.where(g[y0:y1, x0:x1] < 160)
            if len(xs): b = [x0 + xs.min() - 2, y0 + ys.min() - 2, x0 + xs.max() + 3, y0 + ys.max() + 3]
        r = T.count_symbol(img, b, thresh=0.5 if txt else 0.75, scales=(0.9, 1.0, 1.1), rotations=not txt, verify_text=txt)
        ev["boxes"] += [{"box": h["box"], "label": txt or ""} for h in r["hits"]]
        return r["count"], f"example at {[int(v) for v in b]}", ev
    if tool == "measure_area":
        if not ppm: return None, "no scale found", ev
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (g < 170).astype(np.uint8); L = int(0.008 * max(H, W))
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        tot = np.zeros((H, W), bool)
        for s in a.get("seeds", []):
            X, Y = [int(v) for v in px(s, W, H)]
            if lines[Y, X]:                                 # seed on a line: step to the nearest free pixel
                ys, xs = np.where(lines[max(0, Y - 60):Y + 60, max(0, X - 60):X + 60] == 0)
                if not len(xs): continue
                j = np.argmin((ys - min(Y, 60)) ** 2 + (xs - min(X, 60)) ** 2); X, Y = max(0, X - 60) + xs[j], max(0, Y - 60) + ys[j]
            free = (1 - lines) * 255; m = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(free, m, (int(X), int(Y)), 128)
            reg = free == 128
            if reg.mean() < 0.5: tot |= reg                  # ignore a seed that leaked into the whole sheet
        if not tot.any(): return None, "every seed leaked out of its region (open doorways / gaps in the walls)", ev
        tot = cv2.dilate(tot.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        area_m2 = tot.sum() / ppm ** 2
        ev["mask"] = tot
        return (area_m2 / 0.092903 if a.get("unit") == "ft2" else area_m2), f"{len(a.get('seeds', []))} regions", ev
    if tool == "measure_length":
        if not ppm: return None, "no scale found", ev
        pts = [px(p, W, H) for p in a.get("points", [])]
        ev["lines"].append(pts)
        return T.polyline_length(pts, ppm), f"{len(pts)} points", ev
    if tool == "from_dimensions":
        texts = sheet_text(img)
        printed = {re.sub(r"[^\d.]", "", t["text"]) for t in texts}
        vals = [str(v) for v in a.get("values", [])]
        ok = [v for v in vals if re.sub(r"[^\d.]", "", v) in printed]
        for t in texts:
            if re.sub(r"[^\d.]", "", t["text"]) in {re.sub(r"[^\d.]", "", v) for v in ok}:
                ev["boxes"].append({"box": t["box"], "label": t["text"]})
        expr = a.get("expression", "")
        if not re.fullmatch(r"[\d.\se+\-*/()]+", expr): return None, f"unsafe expression {expr!r}", ev
        val = float(eval(expr, {"__builtins__": {}}, {}))
        return val, f"{expr} | printed values confirmed by OCR: {ok} of {vals}", ev
    if tool == "read_value":
        b = box_px(a["box"], W, H); x0, y0, x1, y1 = [int(v) for v in b]
        pad = 40; crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        texts = T.read_text(cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), tile=4000, step=4000)
        found = " ".join(t["text"] for t in texts)
        want = str(a.get("value", ""))
        toks = lambda x: set(re.findall(r"[A-Z]+|\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d)", "", x.upper())))
        ok = toks(want) <= toks(found) if toks(want) else False      # every word/number of the value is printed there (any order)
        where = b
        if not ok:                                         # the model's box was off: look for the value anywhere on the sheet
            for t in sheet_text(img):
                if toks(want) and toks(want) <= toks(t["text"]): ok, where = True, t["box"]; break
        ev["boxes"].append({"box": where, "label": "printed" + (" (OCR confirmed)" if ok else " (OCR could not confirm)")})
        v = num(want)
        return (v if v is not None else want), ("OCR confirmed" if ok else f"OCR read: {found[:80]}"), ev
    return None, f"unknown tool {tool}", ev


def num(x):
    """First number in x; understands 26,562 and feet-inches like 323'-4"."""
    if isinstance(x, (int, float)): return float(x)
    t = str(x).replace("\u2019", "'").replace("\u201d", '"')
    m = re.search(r"(\d+)\s*'\s*-?\s*(\d+(?:\.\d+)?)?\s*\"?", t)
    if m and "'" in t: return float(m.group(1)) + (float(m.group(2)) / 12 if m.group(2) else 0)
    m = re.search(r"-?\d[\d,]*\.?\d*", t)
    return float(m.group(0).replace(",", "")) if m else None


def agree(a, b):
    try:
        a, b = num(a), num(b)
        return abs(a - b) <= max(0.02 * abs(b), 0.5 if abs(b) < 50 else 0)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def room_by_fill(img, box, ppm):
    """Room containing the label at `box`, from the drawing's own room fill colour (BIM/CAD exports paint every room).
    Small holes (text, symbols) count as room; enclosed other rooms do not. Returns (area_m2, mask) or (None, None)."""
    from scipy import ndimage
    a = img.astype(int); H, W = a.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    win = a[max(0, cy - 40):cy + 40, max(0, cx - 80):cx + 80].reshape(-1, 3)
    light = win[(win.mean(1) > 200)]
    if len(light) < 50: return None, None
    fill = np.median(light, 0)
    white = fill.min() > 250
    m = (np.abs(a - fill).max(2) <= 8).astype(np.uint8)
    e = 3 if white else 0                                  # white rooms: cut at narrow openings (doorways) first
    me = cv2.erode(m, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e else m
    n, cc = cv2.connectedComponents(me, connectivity=4)
    sub = cc[max(0, cy - 40):cy + 40, max(0, cx - 80):cx + 80]
    if not (sub > 0).any(): return None, None
    k = np.bincount(sub[sub > 0]).argmax()
    reg = cc == k
    if e: reg = cv2.dilate(reg.astype(np.uint8), np.ones((2 * e + 1, 2 * e + 1), np.uint8)).astype(bool) & (m > 0)
    if reg.mean() > 0.4: return None, None                 # leaked into the sheet background
    holes = ndimage.binary_fill_holes(reg) & ~reg
    lab, nh = ndimage.label(holes)
    if nh:
        sizes = ndimage.sum(holes, lab, range(1, nh + 1))
        reg |= np.isin(lab, [i + 1 for i, sz in enumerate(sizes) if sz / ppm ** 2 < 1.0])
    reg = cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # half of each boundary line
    return reg.sum() / ppm ** 2, reg


_GENERIC = {"RUM", "NR", "ROOM", "RAUM", "RUIMTE", "RUUM", "LOCAL", "NO", "THE"}


def find_label(texts, want):
    """OCR text boxes for a room name like 'Gang (Rum nr. 201)' / 'ESIK (18-2)' / '1.02', best match first.
    The room's own number decides (whole-number match, so 201 never matches 203); the name, on the same line or the line
    next to it, breaks ties. If the number is given but not found anywhere, return [] rather than guess between
    rooms that share a name (several 'ESIK' / 'Kontor' on one sheet)."""
    W_ = want.strip().upper()
    norm = lambda s: re.sub(r"[^0-9A-ZÀ-ÿ]", "", s.upper())
    nums = re.findall(r"\d+(?:[.\-/]\d+)*[A-Z]?", W_)
    words = [w for w in re.findall(r"[^\W\d_]{3,}", W_) if w not in _GENERIC]
    has = lambda txt, n: re.search(r"(?<![\d.\-])" + re.escape(n) + r"(?![\d])", txt.upper()) is not None
    def near(t):
        x0, y0, x1, y1 = t["box"]; h = max(8, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return [o for o in texts if o is not t and abs((o["box"][1] + o["box"][3]) / 2 - cy) < 2.5 * h
                and abs((o["box"][0] + o["box"][2]) / 2 - cx) < max(x1 - x0, o["box"][2] - o["box"][0]) + 2 * h]
    scored = []
    for t in texts:
        txt = t["text"]
        if norm(txt) == norm(W_): scored.append((100, t)); continue
        own_num = any(has(txt, n) for n in nums); own_word = any(w in txt.upper() for w in words)
        if not (own_num or own_word): continue
        nb = near(t) if (nums and words) else []
        sc = 10 * own_num + 3 * own_word
        sc += 5 * any(has(o["text"], n) for o in nb for n in nums) * (not own_num)
        sc += 3 * any(w in o["text"].upper() for o in nb for w in words) * (not own_word)
        if nums and sc < 5: continue                       # name only, and the number is nowhere near: ambiguous
        scored.append((sc, t))
    scored.sort(key=lambda x: -x[0])
    if not scored: return []
    top = [t for s_, t in scored if s_ == scored[0][0]]
    return top if len(top) == 1 or scored[0][0] >= 10 else []   # several equally good name-only hits: refuse


def decide(img, plan, val, detail):
    """Which answer to give when the tool and the model disagree. Rules measured on our benchmarks:
    - verified deterministic tools (tags, symbol + text check, printed values, printed dimensions) win: 14/15 on Stoa's sheets
    - the trained floor-plan models win only on clean, large CAD plans like their training data (8/12 vs 4/12 on Swiss);
      on small or scanned plans they fail (2/24 on CubiCasa) and the model's answer is used instead (17/24)."""
    direct, tool = plan.get("direct_answer"), plan.get("tool")
    if val is None: return direct, "model (tool gave no answer)"
    if agree(val, direct): return val, "tool + model agree"
    if tool in ("floorplan_count", "floorplan_room_area"):
        if max(img.shape[:2]) < 1500: return direct, "model (plan too small / scanned for the trained floor-plan models)"
        return val, "trained floor-plan model (clean CAD plan)"
    if tool == "count_tags" and not val: return direct, "model (no tags found on the sheet)"
    if tool == "count_symbol" and not val: return direct, "model (no symbol copies found)"
    if tool == "read_value" and "confirmed" not in str(detail): return direct, "model (printed value not confirmed by OCR)"
    return val, "deterministic tool"


def draw(img, ev, title, out):
    o = img.copy()
    if ev.get("mask") is not None:
        m = ev["mask"]; o[m] = (0.55 * o[m] + 0.45 * np.array([255, 140, 0])).astype(np.uint8)
    for pl in ev.get("lines", []):
        cv2.polylines(o, [np.int32(pl)], False, (0, 0, 255), 8)
    for b in ev.get("boxes", []):
        x0, y0, x1, y1 = [int(v) for v in b["box"]]
        cv2.rectangle(o, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 230), 5)
        if b.get("label"): cv2.putText(o, b["label"], (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 230), 3)
    fs = max(1.0, o.shape[1] / 2600)
    cv2.rectangle(o, (0, 0), (o.shape[1], int(70 * fs)), (255, 255, 255), -1)
    cv2.putText(o, title, (20, int(52 * fs)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * fs, (0, 0, 0), max(2, int(3 * fs)))
    cv2.imwrite(out, o)


RELIABILITY = {          # measured on our benchmarks; higher = trusted more when methods disagree
    "read_value": 6, "from_dimensions": 5, "count_tags": 4, "count_symbol_text": 4,
    "floorplan_large": 3, "count_symbol": 1, "measure_area": 1, "measure_length": 2, "floorplan_small": 0}


def _rank(img, call, val, detail):
    t = call.get("tool")
    if val is None: return -1
    if t in ("measure_area", "measure_length", "floorplan_room_area") and (num(val) or 0) <= 0: return -1   # empty region
    if t == "read_value": return RELIABILITY[t] if "confirmed" in str(detail) and "not" not in str(detail) else 1
    if t == "from_dimensions": return RELIABILITY[t] if "confirmed by OCR" in str(detail) and "[]" not in str(detail).split("confirmed by OCR:")[-1][:4] else 1
    code = lambda s: bool(s) and len(re.sub(r"[^A-Za-z]", "", str(s))) <= 5     # a tag code (PG, FSCV, PS), not a word (ZIMMER, MAKUUHUONE)
    if t == "count_tags":
        if not val: return -1
        if re.search(r"\b(rooms?|spaces?|locales?|ruimtes?|r[äa]ume?|ruumi|tilat?|huone)", str(call.get("_question", "")), re.I):
            return 1                                               # a room count is not a count of one tag prefix
        return RELIABILITY[t] if code((call.get("args") or {}).get("prefix")) else 1
    if t == "count_symbol":
        txt = (call.get("args") or {}).get("text")
        if txt: return (RELIABILITY["count_symbol_text"] if code(txt) else 1) if val else -1
        # plain template matching: on clean CAD it confirms other tools, on scans/raster it finds only the example (1)
        # or every wall junction (76 toilets): never strong enough to overrule the model on its own
        return RELIABILITY[t] if val and val > 1 else -1
    if t == "floorplan_count" and not val: return -1                  # zero detections: the detector does not know this drawing style
    if t in ("floorplan_count", "floorplan_room_area"):
        if "room fill" in str(detail): return 4                     # BIM fill-colour measurement (0.3% on Molio)
        return RELIABILITY["floorplan_large"] if max(img.shape[:2]) >= 1500 else RELIABILITY["floorplan_small"]
    return RELIABILITY.get(t, 1)


def solve(img, plan):
    """Run every proposed tool; answer = value where independent methods agree, else the most reliable method,
    else the model's own answer. Returns a result dict + the evidence of the chosen method."""
    calls = plan.get("calls") or ([{"tool": plan["tool"], "args": plan.get("args")}] if plan.get("tool") else [])
    runs = []
    for c in calls[:3]:
        try:
            v, d, ev = run_tool(img, dict(c, scale_bar=plan.get("scale_bar")))
        except Exception as e:
            v, d, ev = None, f"error: {e!r}"[:200], {"boxes": [], "mask": None, "lines": []}
        runs.append(dict(tool=c.get("tool"), args=c.get("args"), value=v, detail=d, ev=ev,
                         rank=_rank(img, dict(c, _question=plan.get("_question", "")), v, d)))
    direct = plan.get("direct_answer")
    ok = [r for r in runs if r["rank"] >= 0 and r["value"] is not None]
    for i, r in enumerate(ok):      # a later call of the same tool with other args that disagrees is the model hedging:
        r["alt"] = any(o["tool"] == r["tool"] and not agree(o["value"], r["value"]) for o in ok[:i])   # its first call counts
    best = None
    strong = [r for r in ok if r["rank"] >= 2]
    for r in sorted(strong, key=lambda r: -r["rank"]):          # two independent tools agree (weak ones can't vouch for each other)
        if sum(agree(r["value"], o["value"]) for o in strong if o is not r) >= 1:
            best, source, status = r, "tools agree: " + ", ".join(o["tool"] for o in strong if agree(r["value"], o["value"])), "confident"
            break
    if best is None:
        for r in sorted(ok, key=lambda r: -r["rank"]):          # a tool agrees with the model
            if not r["alt"] and agree(r["value"], direct): best, source, status = r, f"{r['tool']} + model agree", "confident"; break
    if best is None and any(r["rank"] >= 2 and not r["alt"] for r in ok):
        best = max((r for r in ok if not r["alt"]), key=lambda r: r["rank"]); source, status = f"most reliable tool ({best['tool']})", "check"
    if best is None:
        return dict(answer=direct, source="model (no reliable tool result)", status="check", runs=runs), {"boxes": [], "mask": None, "lines": []}
    return dict(answer=best["value"], source=source, status=status, runs=runs), best["ev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--plan", default=""); ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    t0 = time.time()
    plan = json.load(open(a.plan)) if a.plan else plan_with_gemini(img, a.question, a.model)
    t1 = time.time()
    r, ev = solve(img, plan)
    t2 = time.time()
    direct = plan.get("direct_answer"); final, source, status = r["answer"], r["source"], r["status"]
    shown = round(final, 2) if isinstance(final, float) else final
    unit = plan.get("answer_unit") or ""
    res = dict(question=a.question, answer=shown, source=source, status=status, unit=unit, model_direct_answer=direct,
               methods=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items() if k != "ev"} for x in r["runs"]],
               reasoning=plan.get("reasoning"), seconds=dict(plan=round(t1 - t0, 1), tools=round(t2 - t1, 1)))
    out = a.out or os.path.splitext(os.path.basename(a.image))[0] + "_answer.jpg"
    draw(img, ev, f"{a.question}  ->  {shown} {unit}   [{status}]", out)
    res["overlay"] = out
    js = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    open(os.path.splitext(out)[0] + ".json", "w").write(js)          # keep every answer next to its overlay
    print(js)


if __name__ == "__main__":
    main()
