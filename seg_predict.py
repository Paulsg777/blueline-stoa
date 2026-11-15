"""Run the two specialist models on plans; write rooms (polygons) and items (points) per plan.
    python seg_predict.py --unet /vol/ckpt/unet.pt --yolo /vol/ckpt/yolo/weights/best.pt --images ... --plans-file ... --key eval --out /vol/seg_preds
"""
import argparse, json, os
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from PIL import Image
from seg_data import upfactor

CLASSES = ["door", "window", "toilet", "washbasin", "bathtub", "shower", "stove", "sink"]


def predict_mask(model, gray, dev, tile=1024, ov=128):
    H, W = gray.shape; acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    step = tile - ov
    ys = list(range(0, max(1, H - tile), step)) + [max(0, H - tile)]; xs = list(range(0, max(1, W - tile), step)) + [max(0, W - tile)]
    with torch.no_grad():
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                c = gray[y:y + tile, x:x + tile].astype(np.float32) / 255.0
                h, w = c.shape; ph, pw = (-h) % 32, (-w) % 32
                c = np.pad(c, ((0, ph), (0, pw)), constant_values=1.0)
                p = model(torch.from_numpy(c)[None, None].to(dev)).float().sigmoid().cpu().numpy()[0, 0][:h, :w]
                acc[y:y + h, x:x + w] += p; cnt[y:y + h, x:x + w] += 1
    return acc / np.maximum(cnt, 1)


def rooms_from_mask(prob, min_px=200):
    m = (prob > 0.5).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, 4)
    rooms = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_px: continue
        one = (cc == i).astype(np.uint8)
        one = cv2.dilate(one, np.ones((3, 3), np.uint8))           # undo the 1-px shrink used in training
        cs, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        filled = np.zeros_like(one); cv2.drawContours(filled, [c], -1, 1, -1)
        poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).tolist()
        if len(poly) >= 3:
            rooms.append({"polygon": poly, "area_px": int(filled.sum())})
    return rooms


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def nms(items, thr):
    keep = []
    for i in sorted(items, key=lambda i: -i["conf"]):
        if not any(k["type"] == i["type"] and _iou(k["box"], i["box"]) > thr for k in keep): keep.append(i)
    return keep


def predict_items(yolo, img, conf=0.05, tile=640, stride=480):   # keep low-confidence boxes; the cut-off is chosen at scoring time
    W, H = img.size
    st = lambda L: [0] if L <= tile else sorted(set(list(range(0, L - tile, stride)) + [L - tile]))
    boxes = [(x, y) for y in st(H) for x in st(W)]
    crops = [img.crop((x, y, x + tile, y + tile)) for x, y in boxes]
    res = yolo.predict(crops, imgsz=tile, conf=conf, iou=0.95, verbose=False)   # almost no suppression here; overlap is resolved at scoring time
    centres = [(x + tile / 2, y + tile / 2) for x, y in boxes]
    items = []
    for ti, ((x, y), r) in enumerate(zip(boxes, res)):
        for b, c, s in zip(r.boxes.xywh.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            px, py = x + float(b[0]), y + float(b[1]); bw, bh = float(b[2]), float(b[3])
            own = min(range(len(centres)), key=lambda k: (centres[k][0] - px) ** 2 + (centres[k][1] - py) ** 2)
            if own == ti:                                           # count each item from the tile that owns its location
                items.append({"type": CLASSES[int(c)], "point": [px, py], "conf": float(s),
                              "box": [px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2]})
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet"); ap.add_argument("--yolo"); ap.add_argument("--images", required=True)
    ap.add_argument("--plans-file", required=True); ap.add_argument("--key", default="eval"); ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cpu" if a.cpu else "cuda"
    os.makedirs(a.out, exist_ok=True)
    for path in (a.unet, a.yolo):                      # fail loudly instead of silently predicting nothing
        if path and path != "none" and not os.path.exists(path):
            raise SystemExit(f"model weights not found: {path}")
    unet = None
    if a.unet and os.path.exists(a.unet):
        unet = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1).to(dev)
        unet.load_state_dict(torch.load(a.unet, map_location=dev)); unet.eval()
    yolo = None
    if a.yolo and os.path.exists(a.yolo):
        from ultralytics import YOLO
        yolo = YOLO(a.yolo)
    plans = json.load(open(a.plans_file))[a.key]
    if a.limit: plans = plans[:a.limit]
    import time
    for p in plans:
        ip = next((os.path.join(d, p + ".png") for d in a.images.split(os.pathsep) if os.path.exists(os.path.join(d, p + ".png"))), None)
        if ip is None: continue
        t0 = time.time(); img = Image.open(ip).convert("RGB"); out = {"plan": p}
        f = upfactor(*img.size)                                        # same upscaling as in training (small images)
        if f > 1: img = img.resize((img.width * f, img.height * f), Image.BICUBIC)
        if unet is not None:
            out["rooms"] = rooms_from_mask(predict_mask(unet, np.asarray(img.convert("L")), dev), min_px=200 * f * f)
            for r in out["rooms"]:
                r["polygon"] = [[x / f, y / f] for x, y in r["polygon"]]; r["area_px"] = r["area_px"] / (f * f)
        if yolo is not None:
            out["items_all"] = predict_items(yolo, img)                 # nearly unsuppressed (double doors survive)
            out["items"] = nms(out["items_all"], 0.7)                    # standard suppression, as before
            if f > 1:
                for it in out["items_all"]:                                  # 'items' holds the same dicts
                    it["point"] = [it["point"][0] / f, it["point"][1] / f]; it["box"] = [v / f for v in it["box"]]
        out["seconds"] = round(time.time() - t0, 2)
        json.dump(out, open(os.path.join(a.out, p + ".json"), "w"))
        print(p, len(out.get("rooms", [])), "rooms", len(out.get("items", [])), "items", out["seconds"], "s", flush=True)
