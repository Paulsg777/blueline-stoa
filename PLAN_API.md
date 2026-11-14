# Question planning API (for our model server)

Blueline answers simple questions with fixed rules. Anything with extra conditions ("doors in the bedrooms", "rooms over 15 m²", "windows and doors") goes to the model server, which turns the question into a **plan**. The page runs the plan in code on the schema it already has, marks every item, and shows "Read as: …". The model never answers with a number itself.

## Endpoint

`POST /plan`, JSON body:

```json
{
  "question": "How many doors open into a bedroom on floor 2?",
  "vocabulary": {
    "itemKinds": ["door","window","toilet","sink","kitchen_sink","shower","bathtub","sauna_stove","other"],
    "roomTypes": ["Living","Bedroom","Kitchen","Bathroom","WC","Sauna","Hall","Corridor","Balcony","Storage","..."],
    "labels": ["MH","OH","KH/WC","..."],
    "apartments": ["A","B"],
    "floors": 5
  }
}
```

Response:

```json
{ "plan": { ... } | null, "modelCalls": 1, "tokens": 212 }
```

Return `"plan": null` when the question can't be expressed as a plan; the page then tries `POST /ask`.

## Plan format

| Field | Meaning |
|---|---|
| `find` | `"item"` or `"room"` (required) |
| `kind` | item kind or list of kinds, from `itemKinds` (for `find: "item"`) |
| `type` | room type or list, from `roomTypes` (for `find: "room"`) |
| `label` | exact printed room label (for `find: "room"`) |
| `in` | where it must be: `roomType`, `roomLabel`, `roomNumber`, `apartment` |
| `has` | rooms that contain items: `{ "kind": "window", "min": 1 }` |
| `where` | area conditions: `[{ "field": "area", "op": ">", "value": 15 }]` (`>`, `>=`, `<`, `<=`, `=`) |
| `do` | `count`, `sum_area`, `max_area`, `min_area` |
| `scope` | `floor` (the one being viewed), `building`, `project` |
| `floor` | a floor number (1 = bottom) or `"top"`; overrides `scope` |
| `readAs` | one plain sentence saying how the question was read; shown to the user |

## Examples

```json
{"find":"item","kind":"door","in":{"roomType":"Bedroom"},"do":"count","readAs":"count doors that open into a bedroom, on this floor"}
{"find":"room","where":[{"field":"area","op":">","value":15}],"do":"count","readAs":"count rooms over 15 m², on this floor"}
{"find":"room","type":"Bedroom","has":{"kind":"window"},"do":"count","scope":"building","readAs":"count bedrooms with at least one window, in the whole building"}
{"find":"item","kind":["window","door"],"do":"count","readAs":"count windows and doors together, on this floor"}
{"find":"room","type":"Kitchen","in":{"apartment":"B"},"do":"sum_area","readAs":"total kitchen area in apartment B"}
{"find":"room","do":"max_area","floor":3,"readAs":"the largest room on floor 3"}
```

Same origin as the page, or allow CORS.
