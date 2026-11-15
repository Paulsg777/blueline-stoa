"""Training data for the two specialist vision models, from the DXF ground truth.

  rooms : one PNG mask per plan, 255 = inside a room (each room shrunk by 1 px, so touching rooms stay separate)
  items : YOLO detection data on 640-px tiles at full resolution (one square box per door/window/fixture)

    python seg_data.py --gt /vol/gt_full --images /vol/data/data/swiss_dwellings --split /vol/split_full.json --out /vol/segdata
"""
import argparse, json, os, random
import numpy as np, cv2
from PIL import Image

CLASSES = ["door", "window", "toilet", "washbasin", "bathtub", "shower", "stove", "sink"]
BOX_M = {"door": 0.9, "window": 1.0, "toilet": 0.6, "washbasin": 0.55, "bathtub": 1.0, "shower": 0.9, "stove": 0.6, "sink": 0.6}
TILE, STRIDE = 640, 480


def room_mask(g):
    W, H = g["image_size"]
    m = np.zeros((H, W), np.uint8)
    k = np.ones((3, 3), np.uint8)
    for r in g["rooms"]:
        if len(r["polygon"]) < 3: continue
        one = np.zeros((H, W), np.uint8)
        cv2.fillPoly(one, [np.round(np.array(r["polygon"])).astype(np.int32)], 255)
        m |= cv2.erode(one, k)
    return m


def upfactor(W, H):
    """Small images (CubiCasa here is ~400x600 px) are upscaled so symbols and walls are as big as in the large plans."""
    return max(1, round(2000 / max(W, H))) if max(W, H) < 1200 else 1


def upscale(img, g):
    W, H = img.size; f = upfactor(W, H)
    if f == 1: return img, g, 1
    img = img.resize((W * f, H * f), Image.BICUBIC)
    g = json.loads(json.dumps(g)); g["image_size"] = [W * f, H * f]
    for r in g["rooms"]: r["polygon"] = [[x * f, y * f] for x, y in r["polygon"]]
    for it in g["items"]:
        it["point"] = [it["point"][0] * f, it["point"][1] * f]
        if it.get("bbox"): it["bbox"] = [v * f for v in it["bbox"]]
    if (g.get("scale") or {}).get("px_per_m"): g["scale"]["px_per_m"] *= f
    return img, g, f


def tile_starts(L):
    if L <= TILE: return [0]
    s = list(range(0, L - TILE, STRIDE)); s.append(L - TILE); return s


def yolo_tiles(img, g, out_img, out_lbl, stem, keep_empty=0.15, rng=random):
    W, H = img.size; ppm = g["scale"].get("px_per_m") or 60; n = 0
    for y in tile_starts(H):
        for x in tile_starts(W):
            lines = []
            for it in g["items"]:
                if it["type"] not in CLASSES: continue
                cx, cy = it["point"]
                if x <= cx < x + TILE and y <= cy < y + TILE:
                    if it.get("bbox"):                                   # CubiCasa: real symbol box
                        x0, y0, x1, y1 = it["bbox"]; bw, bh = max(x1 - x0, 6), max(y1 - y0, 6)
                    else:                                                # Swiss: fixed size in metres
                        bw = bh = BOX_M[it["type"]] * ppm
                    lines.append(f"{CLASSES.index(it['type'])} {(cx - x) / TILE:.6f} {(cy - y) / TILE:.6f} {min(bw, TILE) / TILE:.6f} {min(bh, TILE) / TILE:.6f}")
            if not lines and rng.random() > keep_empty: continue
            name = f"{stem}_{x}_{y}"
            img.crop((x, y, x + TILE, y + TILE)).save(os.path.join(out_img, name + ".png"))
            open(os.path.join(out_lbl, name + ".txt"), "w").write("\n".join(lines))
            n += 1
    return n


def one_plan(p, src, out, di, dl):
    gdir, idir = src
    gp, ip = os.path.join(gdir, p + ".json"), os.path.join(idir, p + ".png")
    if not (os.path.exists(gp) and os.path.exists(ip)): return 0
    g = json.load(open(gp)); img = Image.open(ip).convert("RGB")
    img, g, f = upscale(img, g)
    if f > 1: img.save(os.path.join(out, "upscaled", p + ".png"))      # the U-Net trains on the same upscaled image
    cv2.imwrite(os.path.join(out, "masks", p + ".png"), room_mask(g))
    return yolo_tiles(img, g, di, dl, p, rng=random.Random(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt2", default=""); ap.add_argument("--images2", default=""); ap.add_argument("--split2", default="")
    ap.add_argument("--limit2", type=int, default=2500)
    ap.add_argument("--only2", action="store_true", help="train on the second source only (e.g. CubiCasa)")
    a = ap.parse_args()
    sp = json.load(open(a.split))
    train = sp.get("sft", []) + sp.get("exp", []); val = sp.get("holdout", [])[:30]
    if a.limit: train, val = train[:a.limit], val[:max(1, a.limit // 4)]
    rng = random.Random(0)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True); os.makedirs(os.path.join(a.out, "upscaled"), exist_ok=True)
    src = {p: (a.gt, a.images) for p in train + val}
    if a.gt2:                                                            # second style (CubiCasa)
        sp2 = json.load(open(a.split2)); t2 = sp2["cubi_exp"][:a.limit2]
        for p in t2: src[p] = (a.gt2, a.images2)
        v2 = t2[-60:]; t2 = t2[:-60]
        train, val = (t2, v2) if a.only2 else (train + t2, val + v2)
    for part, plans in (("train", train), ("val", val)):
        di, dl = os.path.join(a.out, "yolo", "images", part), os.path.join(a.out, "yolo", "labels", part)
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        from multiprocessing import Pool
        with Pool(16) as pool:
            nt = sum(pool.starmap(one_plan, [(p, src[p], a.out, di, dl) for p in plans]))
        print(f"{part}: {len(plans)} plans, {nt} item tiles", flush=True)
    yaml = f"path: {os.path.join(a.out, 'yolo')}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    open(os.path.join(a.out, "yolo", "data.yaml"), "w").write(yaml)
    json.dump(dict(train=train, val=val, images={p: src[p][1] for p in train + val}), open(os.path.join(a.out, "plans.json"), "w"))
    print("done")
