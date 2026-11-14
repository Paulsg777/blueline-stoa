# Blueline for Stoa

Ask a whole construction project a question and see exactly where every answer comes from.

Junction 2026, Stoa challenge "Make Blueprints Talk". Single self-contained file: open `index.html` in a browser.

Version: **V4, 14 Nov 2026**

## What the interface does

- **Project → buildings → floors.** A project holds several buildings, each with one sheet per floor (8 floors is fine). The sample project has Building A (housing, 8 floors) and Building B (offices, 3 floors).
- **Read once, ask many.** Each sheet is read once into `schema.json`. Every question is then answered in code over those schemas, so extra questions cost nothing.
- **Ask at any level.** Floor, building or whole project, in English or Finnish. Examples: "Which floor has the most saunas?", "How many apartments have a sauna?", "Net floor area of building B?", "Montako ovea kerroksessa 3 on?"
- **Proof everywhere.** Every counted item is numbered on its floor. Per-floor bars and the floor strip jump to the sheet with the marks. Areas show the calculation and are cross-checked against the areas printed in the rooms.
- **No proof, no answer.** Questions the drawings can't support get "can't verify".
- **Review queue.** Items under 80% confidence are counted but flagged. Confirm or reject them and every answer updates.
- **Quantity takeoff.** Counts and net areas per floor and building, export as CSV, schemas as JSON.
- **Accuracy check.** Stoa scores accuracy on one blueprint, so there is a one-sheet check: counts exact, areas within 2%. It ships with a sample key generated with the drawing. Load Stoa's key as JSON (`[{"question": "...", "answer": 14}]`) to score a real sheet.

## Swapping in the real model

The reader is the only part that changes. Click **Reader** in the header and choose **Blueline API**.

```
POST {endpoint}/analyze      multipart: file (PNG/JPG), sheet (JSON: id, building, level)
→ BluelineSchema (JSON)

GET  {endpoint}/health       → 200
```

```ts
type Pt = [number, number];                  // page-image pixels
type Box = [number, number, number, number]; // x0, y0, x1, y1

interface BluelineSchema {
  drawing: { id: string; file: string; width: number; height: number;
             metersPerPixel: number | null;  // null = no scale, areas disabled
             scale?: { method: string; text?: string } };
  rooms: { id: string; type: string; label?: string; unit?: string;   // unit = apartment id
           polygon: Pt[]; printedAreaM2?: number; conf: number }[];
  items: { id: string; kind: "door" | "window" | "toilet" | "sink" | "kitchen_sink"
                           | "shower" | "sauna_stove" | string;
           box: Box; roomIds: string[]; conf: number }[];
  walls: { id: string; points: Pt[] }[];
  meta:  { source: string; version?: string; msRead?: number; costEUR?: number };
}
```

Room `type` values the answer engine knows: Living, Bedroom, Kitchen, Bathroom, Sauna, Hall, Corridor, Stair, Lift, Storage, Technical, Utility, Lobby, WC, Office, Meeting, OpenOffice, Bike, Wash. Anything else becomes Room.

Notes:
- The API must allow CORS from wherever the page is opened.
- "Also send the sample sheets" renders each sample sheet to PNG and sends it, so the whole sample project can be re-read by the real model in one click.
- Offline testing without the API: upload an image together with a `schema.json` of the same name (for example `A_krs3.png` + `A_krs3.json`). The demo reader uses that JSON as its output.
- Floors and buildings are guessed from file names (`A_krs3`, `floor_2`, `B-level1`) and can be corrected before reading.

## Honesty notes

- The sample sheets are generated, so the demo reader is exact by construction. Real accuracy comes from the model behind the API.
- Read costs on the demo reader are labelled as estimates. Answer cost is zero because answers are computed in code.

## Licence

MIT
