# Deckle

Command-line tool to scan tarot cards into a digital deck format used in [Tarot Canvas](https://github.com/arcanaland/tarot-canvas).

Decks are saved in Arcana Land's [Tarot Deck format](https://github.com/arcanaland/specifications) v2.

## Requirements

- `uv` and `just`.
- A flatbed scanner.
- Access to a 3D printer to create jigs that will hold cards in your scanner.
- A slab of dark material to place on top of the cards while scanning, such as black polyurethane foam (Optional)

## Technical Details

Detecting a card in an image:
- Send a bunch of scanlines across the image to find card edges
- Fit a line through where we think a card edge is

## Reading a detection failure

When a scan fails, `deckle detect` and `deckle rectify` still measure every window and
every edge, then list each problem:

```
deckle: scan.png: 2 problems in 2 of 4 windows
  window 2 (r1c0): right: 400 of 2000 scanlines hit (20%, need 80%) — no_void 1590, void_after 10
  window 3 (r1c1): right: 200 of 2000 scanlines hit (10%, need 80%) — no_void 1760, weak_step 40
```

Each scanline walks inward from the window wall, through the search band, and ends with
one reason. The counts say which rule each scanline stopped at. They describe the rule,
not what caused the miss. Constants are in `src/deckle/edges.py`.

| Reason | Strategy | What the rule was looking for |
|---|---|---|
| `hit` | all | An edge was found and passed every check below. |
| `off_image` | all | A scanline that stays inside the scan. It would have run off the image. |
| `no_room` | all | Enough samples to search the band, or to check what lies past the chosen step. |
| `no_void` | outermost | A run of near-black (luma ≤ `VOID_CEIL`) at least `VOID_RUN_MM` long between the wall and the band end. |
| `weak_step` | brightest, outermost | A rise of at least `STEP_FLOOR` luma at the chosen place. The largest one found was smaller. |
| `no_step` | innermost | Any step of at least `STEP_FLOOR` luma, rising or falling, in the band. |
| `void_after` | outermost | Card for `SUSTAIN_MM` past the step. Near-black came back instead. |
| `not_flat` | brightest, innermost | A level surface for `SUSTAIN_MM` past the step. Its luma varied by more than 12 (standard deviation). |
