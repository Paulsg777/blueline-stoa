# Blueline for Stoa

Ask a whole construction project a question and see exactly where every answer comes from.

Junction 2026, Stoa challenge "Make Blueprints Talk". Single self-contained file: open `index.html` in a browser.

Version: **V6.1, 14 Nov 2026**

## What the interface does

The interface does not read drawings itself. A reader turns each floor plan into a **schema** (rooms, doors, windows, fixtures, scale). Everything after that is the interface: plan overlay, 3D, answers with proof, and the trace.

1. **Add floor plans.** One image per floor, or a PDF with a page per floor. Each floor becomes a card, ordered by the number in the file name (`krs1`, `floor-2`...). Drag to reorder, drop on a building tab to move, double-click to rename, delete has undo.
2. **Reading, in parallel.** Floors are read at the same time: 3 with our model server, 2 on a claude.ai account. A rate limit drops it to one at a time; there is no retry loop.
3. **Plan and 3D.** Plan shows one floor with what was read in blue. 3D stacks the building and is built from the schema alone (see below).
4. **Ask.** About a floor, the building, or all buildings. Answers are plain code over the schemas, so asking makes no model calls. Every counted item is numbered on its floor.
5. **Trace (T).** A panel under the viewer shows every step, live:
   - **Reading:** a timeline of floors being read in parallel, then for each floor: the file, image preparation, the reader's call with live output and tokens, parsing, outline snapping, scale detection, linking items to rooms, and the 3D build.
   - **Answer:** how the question was read (which rule, which words matched, which language), which floors were used and why, each tool call with the items behind it, the sum, the check (printed vs traced areas), and the cost.

## Readers (Settings)

| Reader | Use | Claude needed |
|---|---|---|
| **Our model** | The team's model server. `POST /api/plan` (raw image, `X-Filename`), then `GET /api/scene?id=` | No |
| **Schema files** | Drop each plan image with its schema JSON of the same name (`plan_krs1.png` + `plan_krs1.json`), or the JSON alone | No |
| **Claude** | Stand-in until our model is ready. Runs only inside claude.ai, on your own Claude account | Yes, no API key |

Schema files can be in either format:
- the team's schematic format (`schema.md` in reinforcement-testing): `rooms[].polygon`, `number`, `label`, `printed_area_m2`, `apartment`, `items[].type` with a `point`, `scale.px_per_m`, `apartments[]`
- the Blueline contract below

A server with `POST /analyze` that returns the Blueline contract also works; the reader falls back to it when `/api/plan` is missing.

### Connecting the model server

Inside claude.ai the page cannot call other servers. Two options:

1. **Serve this page from the model server** (simplest). Copy `index.html` into the server's `web/` folder. The page detects the server on its own origin and uses it.
2. **Allow CORS** in `app.py` and open the page locally:

```python
def _cors(self):
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

def do_OPTIONS(self):
    self.send_response(204); self._cors(); self.end_headers()

# and call self._cors() in _json() just before self.end_headers()
```

When the server returns a 3D scene (`elements` with `box {c, s, yaw}`), the 3D view draws that scene as delivered.

## 3D from the schema

No image processing. Rooms are filled on a 4 cm grid, touching rooms get a wall between them, and the band around the rooms becomes wall. Balconies get a railing. Each door or window cuts an opening: boxes use the side that sits on a wall (checked against the door's two rooms), points open a standard width (0.9 m door, 1.2 m window) in the nearest wall. Windows get a sill, a head and glass; fixtures are sized by kind. The trace lists how many openings were placed and which items were not on a wall.

## Answers

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
           polygon: Pt[]; printedAreaM2?: number | null }[];
  items: { id: string; kind: "door" | "window" | "toilet" | "sink" | "kitchen_sink" | "shower" | "bathtub" | "sauna_stove" | string;
           box: Box; pt?: Pt; roomIds: string[] }[];
  meta:  { source: string };
}
```

## Honesty notes

- The sample building is generated, so its "reading" is exact by construction and makes no model call. The trace says so.
- No API key is used or stored. Any key saved by an earlier version is deleted from the browser when the page loads.
- Claude token counts in the trace are estimates.
- Areas are given only where they are printed or the drawing has a scale; otherwise the answer says it cannot verify.

## Licence

MIT
