# BSEARCH frontend ↔ our reader

**Run:** `python3 blueline_api.py` in `floorplan_env` → `http://localhost:8000`. In the page: **Reader → BSEARCH API → `http://localhost:8000`**. Open `index.html` locally. CORS is open, so `file://` works.

The server implements the frontend's contract unchanged. It also adds one endpoint.

## `POST /analyze` (file, sheet) → BSEARCHSchema

This is read deterministically, with **no model call** (`meta.modelCalls = 0`, `meta.costEUR = 0`):

- `drawing.metersPerPixel` comes from the scale bar. Its length is measured in pixels, and its printed numbers are read by OCR. `drawing.scale.box` is where the bar is.
- `rooms[]` are the painted rooms (BIM/CAD exports). Each has a polygon, `type` (mapped from the label in FI/EE/DA/DE/NL/IT/EN), `label`, and `labelBox`.
  - **`measuredAreaM2`** is a new field: the net area measured on the drawing. It is within about 2% of the official value on the plans we checked. Enclosed shafts and rooms are excluded, so this differs from the shoelace area of the outer polygon.
  - `printedAreaM2` is set only when an area is actually printed inside the room.
- `items[]` are doors, windows and fixtures from our trained detectors, when their weights are present. On sheets where no detector applies, the list is empty rather than guessed.
- `texts[]` is every piece of text on the sheet, with boxes.
- **`tags[]`** is a new field: tagged objects (`FSCV-1..7`, `PS-1..24`), with prefix, count, numbers and boxes. Piping sheets are answered from this.

**One-line change needed in index.html** (line ~757, `roomArea`): prefer the measured area.

```js
const roomArea=(sc,r)=>r.measuredAreaM2!=null?+r.measuredAreaM2:r.printedAreaM2!=null?+r.printedAreaM2:sc.drawing.metersPerPixel?shoelace(r.polygon)*sc.drawing.metersPerPixel**2:null;
```

## `POST /ask` (file, question) → answer with evidence

This is for questions the schema engine can't answer, such as printed dimensions, tags, or "how long is the trench". Use it where the page would otherwise say "can't verify".

The order is: schematic first (0 model calls), then Gemini plans local tools, then the model's own answer flagged `check`.

```json
{"answer": 70.07, "unit": "m²", "status": "confident" | "check", "source": "schematic (no model call)",
 "modelCalls": 0, "tokens": 0, "methods": [{"tool", "value", "detail"}],
 "evidence": {"polygons": [[[x,y],...]], "boxes": [[x0,y0,x1,y1],...], "lines": [[[x,y],...]]}}
```

Coordinates are page-image pixels, the same frame as `/analyze`. Draw `evidence` on the plan, and show `status` and `source` next to the answer.
