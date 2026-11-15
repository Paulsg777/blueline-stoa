# Blueline for Stoa

Ask a whole construction project a question and see exactly where every answer comes from.

Junction 2026, Stoa challenge "Make Blueprints Talk". Single self-contained file: open `index.html` in a browser.

Version: **V6.2, 14 Nov 2026**

## What the interface does

Our model reads each floor plan into a **schema** (rooms, doors, windows, fixtures, scale). Everything after that is the interface: plan overlay, 3D, answers with proof, and the trace. There is no Claude and no API key in the page.

1. **Add floor plans.** One image per floor, or a PDF with a page per floor. Each floor becomes a card, ordered by the number in the file name (`krs1`, `floor-2`...). Drag to reorder, drop on a building tab to move, double-click to rename, delete has undo.
2. **Reading, in parallel.** Up to 3 floors are sent to our model at the same time. The original file goes to the model (sharper text for its reader); a smaller copy is used for the views.
3. **Plan and 3D.** Plan shows one floor with what was read in blue. 3D stacks the building and is built from the schema alone (see below).
4. **Ask.** About a floor, the building, or all buildings. Counts and areas are plain code over the schemas, with every item marked. Questions the schema cannot answer (sizes, lengths, tags such as `PS-12`) go to our model's `POST /ask` for the current floor.
5. **Trace (T).** A panel under the viewer shows every step, live:
   - **Reading:** a timeline of floors being read in parallel, then for each floor: the file, the call to our model and what it returned, the scale, linking items to rooms, and the 3D build.
   - **Answer:** how the question was read, which floors were used and why, each tool call with the items behind it, the sum, the check, and the cost. For `/ask` answers: the model's own methods, model calls and tokens.

## Our model

The page finds the model by itself, in this order: `?api=https://...` in the page address, the server that serves the page, then the last address used (Settings). Hosted on Modal, one URL gives both the page and the reader.

| Endpoint | Use |
|---|---|
| `POST /analyze` (multipart: `file`, `sheet`) | Reads one floor into a schema. Our reader (`blueline_api.py`) is deterministic: scale from the scale bar by OCR, rooms with `measuredAreaM2`, room types from labels in 7 languages, texts with `tags[]` |
| `POST /ask` (multipart: `file`, `question`) | Answers what the schema cannot: `answer`, `unit`, `status`, `source`, `methods[]`, `modelCalls`, `tokens` |
| `POST /api/plan` + `GET /api/scene?id=` | The HackToTheFuture server; its 3D scene is drawn as delivered |

The page prefers `measuredAreaM2`, then a printed area, then the outline area at the drawing scale.

**Model output files, no server:** drop each plan image with its schema JSON of the same name (`plan_krs1.png` + `plan_krs1.json`), or the JSON alone. Both the Blueline contract (below) and the schematic format from `schema.md` (rooms with `polygon`, `number`, `label`, `printed_area_m2`, `apartment`; items with a `point`; `scale.px_per_m`) work.

Inside claude.ai the page cannot call other servers, so open it from the model's URL instead. Running the HackToTheFuture server locally needs CORS (or serve `index.html` from its `web/` folder):

```python
def _cors(self):
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

def do_OPTIONS(self):
    self.send_response(204); self._cors(); self.end_headers()

# and call self._cors() in _json() just before self.end_headers()
```

## 3D from the schema

No image processing. Rooms are filled on a 4 cm grid, touching rooms get a wall between them, and the band around the rooms becomes wall. Balconies get a railing. Each door or window cuts an opening: boxes use the side that sits on a wall (checked against the door's two rooms), points open a standard width (0.9 m door, 1.2 m window) in the nearest wall. Windows get a sill, a head and glass; fixtures are sized by kind. The trace lists how many openings were placed and which items were not on a wall.

## Answers

How a question is handled:
1. **Fixed rules, only when the rule is the whole question.** Stoa templates and simple questions ("how many doors on this floor") are answered in code at once. If the question adds conditions the rule can't handle (in, with, per, larger than, not, and...), the rule is not used.
2. **Our model plans the rest** (`POST /plan`, see `PLAN_API.md`): it turns the question into a plan such as "doors, in bedroom rooms, count". The page runs the plan on the schema and marks every item. One small model call; the number itself always comes from what was read.
3. **Sizes, lengths and tags** go to `POST /ask`.
4. Otherwise the page says how it would have read the question and does not guess.

Every answer starts with "Read as: …".


The question rules follow the team's `answer.py`, so the interface answers the way the team scores:
- combined items: sinks and washbasins, showers and bathtubs, toilet bowls
- room types from the printed label first (`KH/WC` is a bathroom, not a WC room), the reader's type as fallback
- rooms labelled "X", area of room 3.09, area of apartment B or all numbered indoor rooms without outdoor spaces, storeys on the sheet
- English, Finnish, German, French and Italian templates

"This floor plan" means the floor you are looking at; "in total" no longer jumps to the whole building.

## Blueline contract

```ts
type Pt = [number, number];                  // page-image pixels
type Box = [number, number, number, number]; // x0, y0, x1, y1

interface BluelineSchema {
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

- The sample building is generated, so its "reading" is exact by construction and makes no model call. The trace says so.
- No Claude and no API key. Any key saved by an earlier version is deleted from the browser when the page loads.
- Areas are given only where they are printed or the drawing has a scale; otherwise the answer says it cannot verify.

## Licence

MIT

## The reader (server)

The files next to `index.html` are our model's server: `POST /analyze`, `/ask`, `/plan`, and the page itself on `GET /`.

How a sheet is read (`blueline_api.py`):
1. **One model call per sheet** (Gemini 4 Argon) says what is there: the sheet type, the rooms with a point and any printed area, doors, windows, fixtures, other equipment with its printed tag, and where the scale bar is. It never measures.
2. **Code measures and checks everything.** Scale from the scale bar by OCR, checked against the bar the model points at, against a printed ratio and for a plausible sheet size; with no bar, from the printed room areas. Room outlines from our segmenter or grown from the drawn walls, whichever lies on the walls (`on_walls`). Doors from our detector, a model-listed door only where code finds its swing arc (`arcs.py`). Equipment boxes on its tag as read by OCR.
3. **A second look** (on by default, `VERIFY=0` turns it off): the model sees our result drawn on the sheet and names mistakes; code checks each claim before anything changes, and every decision is in `meta.check`.

Run it: put `GEMINI_API_KEY=...` in `.env`, then `python3 blueline_api.py` (http://localhost:8000). Host it: `modal deploy blueline_modal.py`.
The trained weights (`unet_swiss.pt`, `yolo_swiss.pt`) are not in this repository; without them the reader still works, using the model's list and the wall-grown rooms.
`inspect_sheet.py <sheet>` draws what the reader found on a sheet; `compare_models.py` compares perception models on sheets with ground truth.
