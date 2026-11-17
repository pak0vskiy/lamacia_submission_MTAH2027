# 2026-11-04 · IDSSE as a source of dead-ball labels

Author: Maxim. Method: `analysis/03_idsse_probe.py` (aggregates only). External public data
(DFL/IDSSE, CC BY 4.0, other games). One match checked so far: J03WPY (2. Bundesliga).

## TL;DR for the team

1. **Yes: IDSSE has a per-frame alive/dead flag and a per-frame owning team**, exposed by kloppy
   as `ball_state` and `ball_owning_team`. Dead-ball labels come straight from the provider's
   tracking, no event-derivation needed. That's the same kind of label as our GT (D = "corrected
   tracking marks ball dead").
2. **Same 25 fps as our ATD.** Windows and durations transfer 1:1.
3. **Gotcha: kloppy's loader defaults to `only_alive=True`**, which silently drops every dead
   frame. Always pass `only_alive=False`.
4. **Gotcha: kloppy's figshare URLs return an empty body to scripts.** Use the HuggingFace mirror
   `pysport/idsse-data` (~370 MB positions XML per match, 7 matches ≈ 2.6 GB).
5. **The data is clean; ours isn't.** Ball present 100 %, exactly 22 players every frame, ball z
   available. To be useful for us it needs ball blackouts in long blocks, player drop-out /
   fragmentation, and no z.

## Numbers (J03WPY)

146,211 frames, 2 periods, 25 fps.

| | period 1 | period 2 |
|---|---|---|
| minutes | 46.1 | 51.4 |
| dead share | 33.2 % | 54.2 % |
| dead spells | 38 | 59 |
| dead-spell duration q10/25/50/75/90, s | 5.9 / 10.2 / 22.5 / 34.1 / 40.1 | 5.2 / 10.5 / 22.8 / 35.5 / 58.6 |
| dead spells < 1 s | 0 | 0 |
| alive-spell duration q10/25/50/75/90, s | 13.3 / 31.4 / 44.1 / 62.4 / 81.3 | 4.2 / 7.9 / 17.3 / 33.6 / 49.1 |
| owner changes while alive | 166 | 158 |
| ball z non-zero | 66.6 % | 45.4 % |

Whole match: ~44 % dead. For reference, Sam's v1 labels 43 % of BAR-ATH graded frames D (no GT
involved on either side; one match, so only a plausibility check).

## Caveats

- One match only. The other six should share the format but are not checked yet.
- **Label conventions may differ from Metrica's GT**: when exactly D starts (ball over the line vs
  whistle) and how DFL defines `BallPossession` vs the "controlled possession" rule. Boundary
  offsets of a second matter for spellF1. Can't be measured on BAR-ATH; at best cross-check DFL
  against the IDSSE event feed (ball-out / set-piece timestamps).
- kloppy maps any `BallPossession` other than 1 to away. Fine while alive; ignore owner while dead.
- Bundesliga 1 / 2 only; a different league and camera setup from BAR-ATH.

## Implications (to discuss)

- **A learned D detector trained on IDSSE is feasible**: 7 matches ≈ 22 h of frames, ~40 % D,
  roughly 600 dead spells.
- Next: fetch the other 6 matches, build the noise model (ball blackouts in long blocks, gap
  durations matched to our q10–q90 of 0.5–33.8 s; randomise how strongly blackouts coincide with D,
  since we can't measure it here), team-level features without ball z, leave-one-match-out CV.
- Document IDSSE (source, licence, how labels were produced) in `REPORT.md`.
