# BSEARCH

**Construction drawings, read once into measured data you can search and check.**

Junction 2026, Stoa challenge "Make Blueprints Talk". Live: https://aistis-verkauskas--bsearch.modal.run

Ask "How many 60 × 60 cm tiles do we need for every bathroom in both buildings?" and BSEARCH answers with one number, every bathroom marked on the floor it came from, and the sum, division and rounding written out in the trace. The language model never estimates the number; code measures it on the drawings.

## Results

On 115 held-out Swiss plans (our models never saw them in training):

| | BSEARCH | Gemini, one call on the whole image |
|---|---|---|
| Room areas within 2% | 79.3% | 38.9% |
| Exact counts (doors, windows, fixtures) | 89.2% | 74.5% |
| Overall score | 0.877 | 0.587 |

Single projects: site1102, all 21 room areas within 2% and every window, toilet, sink, bathtub and stove counted exactly; Molio office building (3 floors), 45 of 47 room areas within 2%; drawing scale within 0.2% of the CAD ground truth.

## The app

One file, `index.html`, served by the reader. Three views:

- **Database.** 607 drawings (CAD floor plans from 14 BIM projects, 500 scans, elevations and sections) shown as a city of projects. Search the archive by room name, number or type ("bathrooms", "Küche", "205"); every hit is marked on its drawing. Click a floor to open it in Plan. "Give it to an AI agent" hands an agent the archive index with instructions to answer from the data, not by guessing.
- **Plan.** One floor with everything that was read drawn over it: rooms, doors, windows, fixtures, the scale. Press O for the original drawing.
- **3D.** Every building side by side, floors stacked, built from the measured data alone (rooms filled on a 4 cm grid, walls between rooms, door and window openings cut where the items sit).

**Adding drawings.** Images, a PDF (one page per floor), or a schema JSON from our reader (loaded with no model call). Floors read in parallel, one building per tab; drag to reorder.

**Asking.** About one floor, a building, or all buildings:
1. Simple template questions ("how many doors on this floor") are answered by fixed rules, but only when the rule understands every word of the question.
2. Everything else goes to a small planning call (`POST /plan`, Gemini 3.8 Flash) that returns a plan; code runs it on the stored data. Plans can count, sum areas, filter by room type, size or item, do arithmetic with units (`then`: tiles, flooring, rounding), and test whether a shape fits in a room (rasterised at 5 cm).
3. Judgement questions ("Is this building wheelchair friendly?") become several measured sub-questions; `POST /conclude` writes a short conclusion that may only cite those answers and names what was not covered.

Every answer starts with "Read as: …", marks each room or item behind it on its drawing (with a button to fly to each building in 3D), and the trace (T) lists every step with its numbers and cost.

## The reader (server)

`blueline_api.py` reads a sheet in four steps:

1. **See.** One Gemini 4 Argon call lists what is on the sheet: sheet type, rooms with a point and any printed area, doors, windows, fixtures, equipment with its printed tag, the scale bar and the building outline. It never measures.
2. **Measure.** Scale from the scale bar by OCR, checked against the bar the model points at, a printed ratio and a plausible sheet size; with no bar, from printed room areas (`schematic.py`, `scale_bar.py`). Room outlines from our U-Net segmenter or grown from the drawn walls, whichever lies on the walls (`on_walls`). Doors from our YOLO detector; a door the model lists is added only where code finds its swing arc (`arcs.py`). Equipment is placed on its tag as read by OCR. A room whose outline can't be traced is kept as a marker without an area.
3. **Check.** The model sees our result drawn over the sheet and names mistakes; code tests each claim before anything changes. Decisions are in `meta.check` and the trace. On by default (`VERIFY=0` turns it off).
4. **Store.** The result is cached per drawing and model, so each drawing is read once.

Endpoints: `POST /analyze` (read a sheet), `POST /plan`, `POST /conclude`, `POST /ask`, `GET /db/*` (archive index), `GET /` (the page).

Cost: about 3,000 tokens per drawing, read once; then one small planning call per question. The same question on the same drawings always gives the same number.

## Run it

```
pip install -r requirements.txt
echo "GEMINI_API_KEY=..." > .env
python3 blueline_api.py          # http://localhost:8000
modal deploy blueline_modal.py   # hosted on Modal
```

The trained weights (`unet_swiss.pt`, `yolo_swiss.pt`, trained on Modal GPUs) are not in this repository; without them the reader still works, using the model's list and wall-grown rooms.

## Repository map

| File | What it is |
|---|---|
| `index.html` | The whole frontend: Database, Plan, 3D, answers, trace |
| `blueline_api.py` | The reader and all endpoints |
| `schematic.py`, `scale_bar.py`, `arcs.py`, `drawing_tools.py` | Scale, room outlines, door arcs, drawing helpers |
| `seg_data.py`, `seg_predict.py`, `seg_post.py`, `ask.py` | U-Net and YOLO data prep, inference, post-processing |
| `blueline_modal.py` | Modal deployment |
| `db/` | Archive index (`index.json`: room names, numbers, types and positions for 607 drawings), thumbnails, `build_index.py` |
| `PLAN_API.md`, `BLUELINE_API.md` | API details |
| `inspect_sheet.py`, `compare_models.py`, `list_models.py` | Debugging: draw what was read; compare perception models |

## BSEARCH contract

```ts
type Pt = [number, number];                  // page-image pixels
type Box = [number, number, number, number]; // x0, y0, x1, y1

interface BSEARCHSchema {
  drawing: { width: number; height: number;
             metersPerPixel: number | null;  // null = no scale, areas only where printed
             scale?: { method: string; text?: string; estimated?: boolean }; storeys?: number };
  rooms: { id: string; type: string; label?: string; number?: string; unit?: string;  // unit = apartment
           polygon: Pt[]; printedAreaM2?: number | null; measuredAreaM2?: number | null }[];
  items: { id: string; kind: "door" | "window" | "toilet" | "sink" | "kitchen_sink" | "shower" | "bathtub" | "sauna_stove" | string;
           box: Box; pt?: Pt; roomIds: string[] }[];
  meta:  { source: string };
}
```

## Honesty notes

- The archive search uses room labels from each drawing's text layer. Areas and counts are measured when a drawing is opened and read.
- The sample building is generated, so its reading is exact by construction and makes no model call. The trace says so.
- Areas are given only where a scale is found or an area is printed; otherwise the answer says it cannot verify.

## Licence

MIT
