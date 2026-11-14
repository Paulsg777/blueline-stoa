# Blueline for Stoa

Ask a whole construction project a question and see exactly where every answer comes from.

Junction 2026, Stoa challenge "Make Blueprints Talk". Single self-contained file: open `index.html` in a browser.

Version: **V5, 14 Nov 2026**

## How it works

1. **Add floor plans.** One image per floor (PNG, JPG, SVG), or a PDF with a page per floor. Each floor becomes a card in the left panel, ordered by the number in the file name (`krs1`, `floor-2`...).
2. **Put them in order.** Drag cards up or down; the top card is the top floor and floor numbers update. Drop a card on a building tab to move it to that building. Double-click a name to rename it. Delete has undo. Keyboard: Alt + arrows moves a floor.
3. **Reading.** Each floor is read once into `schema.json` (rooms, doors, windows, fixtures, scale). For the trial this is Claude: inside claude.ai it runs on the viewer's own Claude account; in a downloaded copy it uses an Anthropic API key kept in the browser. The trace appears live while Claude reads.
4. **Plan and 3D.** Plan shows one floor with what was read in blue. 3D stacks the whole building: plates show each drawing, walls rise as each floor is read. Click a floor to pick it, double-click to open its plan. Spread pulls floors apart; "Only this floor" isolates one.
5. **Ask.** About this floor, the building, or all buildings, in English or Finnish. Answers are computed in code from the schemas, so extra questions cost nothing. Every counted item is numbered on its floor, per-floor bars jump to that floor, and in 3D each counted item gets a pin. Questions the drawings cannot support get "can't verify".

Sample building: one click loads a generated 5-floor building to try everything without uploads.

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
