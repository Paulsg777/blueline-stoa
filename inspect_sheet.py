"""See what the reader finds on a sheet: exactly what /analyze returns, drawn on the drawing.

    python3 inspect_sheet.py examples/swiss_1040.png            (any PNG/JPG/PDF; several files allowed)
    -> inspect_<name>.jpg opened in Preview + a summary in the terminal

Blue outlines = rooms (label, type, measured m²) · coloured boxes = detected doors / windows / fixtures ·
red box = the scale bar it used · magenta = tagged objects (PS-12, FSCV-3 ...).
"""
import collections, os, subprocess, sys, time
import numpy as np, cv2
import blueline_api as B, schematic as S, ask

COL = {"door": (0, 140, 255), "window": (255, 160, 0), "toilet": (0, 200, 0), "sink": (200, 0, 200), "kitchen_sink": (120, 0, 200),
       "shower": (0, 200, 200), "bathtub": (0, 120, 120), "sauna_stove": (0, 0, 160)}

for path in sys.argv[1:]:
    img = B.load_image(open(path, "rb").read()); t0 = time.time()
    sch = S.build(img, key=ask._img_key(img))
    o = B.to_blueline(sch, img, {}, os.path.basename(path), 0)
    secs = time.time() - t0
    v = img.copy(); H, W = v.shape[:2]; fs = max(0.5, W / 3500); lw = max(2, int(W / 1200))
    for r in o["rooms"]:
        cv2.polylines(v, [np.int32(r["polygon"])], True, (255, 90, 0), lw + 1)
        x, y = np.int32(r["polygon"]).mean(0).astype(int)
        txt = f"{r['type']}: {r['label'][:22]}" + (f"  {r['measuredAreaM2']:.1f} m2" if r.get("measuredAreaM2") else "")
        cv2.putText(v, txt, (int(x) - 60, int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 60, 0), max(1, lw - 1))
    for it in o["items"]:
        b = it["box"]; c = COL.get(it["kind"], (80, 80, 80))
        cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), c, lw)
        cv2.putText(v, it["kind"], (b[0], b[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, c, 1)
    sc = o["drawing"]["scale"]
    if sc.get("box"): b = sc["box"]; cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), (0, 0, 230), lw + 2)
    for t in o.get("tags", []):
        for b in t["boxes"]: cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), (200, 0, 200), lw)
    head = (f"scale: {sc.get('text') or 'NONE'} | rooms: {len(o['rooms'])} ({sch.get('rooms_source')}) | "
            + ", ".join(f"{k} {n}" for k, n in collections.Counter(i['kind'] for i in o['items']).most_common()) + f" | {secs:.0f} s, {o['meta'].get('modelCalls', 0)} model call(s), {o['meta'].get('tokens', 0)} tokens")
    cv2.rectangle(v, (0, 0), (W, int(60 * fs)), (255, 255, 255), -1)
    cv2.putText(v, head, (10, int(45 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (0, 0, 0), max(1, lw))
    out = "inspect_" + os.path.splitext(os.path.basename(path))[0] + ".jpg"
    cv2.imwrite(out, v)
    print(f"\n{path}\n  {head}")
    for r in o["rooms"]: print(f"   room {r['type']:10s} {r['label'][:40]:40s} {r.get('measuredAreaM2')}")
    if sys.platform == "darwin": subprocess.run(["open", out])
