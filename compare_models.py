"""Which perception model reads sheets best? Same sheets, same reader, only the perception model changes.

    python3 compare_models.py <sheet.pdf|png> [more sheets]          (Swiss sheets: ground truth found in gt/ by site number)
    python3 compare_models.py --models gemini-3.8-flash,gemini-4-argon <sheets>

Per model and sheet: rooms whose area is within 2% of the ground truth (matched by printed room number), exact counts
per object type, seconds and tokens of the one perception call. Sheets without ground truth just show the counts.
"""
import argparse, collections, glob, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import blueline_api as B, schematic as S, ask

DEFAULT = "gemini-3.8-flash,gemini-3.1-pro-preview,gemini-4-argon,gemini-4-carbon,gemini-4-hydrogen"
KIND = {"washbasin": "sink"}

ap = argparse.ArgumentParser(); ap.add_argument("sheets", nargs="+"); ap.add_argument("--models", default=DEFAULT)
a = ap.parse_args(); models = a.models.split(",")
if not os.environ.get("GEMINI_API_KEY"): sys.exit("GEMINI_API_KEY not found in floorplan_env/.env")


def gt_for(path):
    m = re.search(r"site(\d+)", os.path.basename(path))
    hits = glob.glob(f"gt/site{m.group(1)}_*.json") if m else []
    return json.load(open(hits[0])) if hits else None


def score(o, gt):
    if not gt: return None
    labels = [(r.get("label") or "", r.get("measuredAreaM2")) for r in o["rooms"]]
    ok = n = 0
    for g in gt["rooms"]:
        num, area = str(g.get("number") or ""), g.get("area_m2")
        if not num or not area: continue
        n += 1
        got = [v for l, v in labels if re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", l) and v]
        ok += bool(got) and abs(got[0] - area) <= 0.02 * area
    truth = collections.Counter(KIND.get(i["type"], i["type"]) for i in gt["items"])
    mine = collections.Counter(i["kind"] for i in o["items"])
    counts = {k: (mine.get(k, 0), t) for k, t in truth.items()}
    return ok, n, counts


summary = collections.defaultdict(lambda: [0, 0, 0, 0])               # model -> rooms ok, rooms, counts exact, counts
for path in a.sheets:
    img = B.load_image(open(path, "rb").read()); gt = gt_for(path)
    sch = S.build(img, key=ask._img_key(img))
    def run(m):                                                      # the perception calls run in parallel
        t0 = time.time()
        try: p = B.perceive(img, model=m); return m, p, time.time() - t0, None
        except Exception as e: return m, None, time.time() - t0, f"{type(e).__name__}: {str(e)[:300]}"
    with ThreadPoolExecutor(len(models)) as ex: res = list(ex.map(run, models))
    print(f"\n{os.path.basename(path)}" + ("" if gt else "   (no ground truth: counts only)"))
    for m, p, secs, err in res:
        if err: print(f"  {m:24s} FAILED  {err}"); continue
        os.environ["PERCEIVE_MODEL"] = m                             # to_blueline now reuses this model's cached read
        o = B.to_blueline(sch, img, {}, os.path.basename(path), 0)
        c = collections.Counter(i["kind"] for i in o["items"])
        line = f"  {m:24s} {secs:5.0f} s {p.get('_tokens', 0):7d} tok | rooms {len(o['rooms']):3d}"
        sc = score(o, gt)
        if sc:
            ok, n, counts = sc; ex_ = sum(x == t for x, t in counts.values())
            s_ = summary[m]; s_[0] += ok; s_[1] += n; s_[2] += ex_; s_[3] += len(counts)
            line += f", {ok}/{n} within 2% | counts exact {ex_}/{len(counts)}: " + ", ".join(f"{k} {x}/{t}" for k, (x, t) in sorted(counts.items()))
        else:
            line += " | " + ", ".join(f"{k} {n}" for k, n in c.most_common())
        print(line, flush=True)
if summary:
    print("\nTOTAL (sheets with ground truth)")
    for m in models:
        if m in summary: s_ = summary[m]; print(f"  {m:24s} rooms within 2%: {s_[0]}/{s_[1]}   exact counts: {s_[2]}/{s_[3]}")
