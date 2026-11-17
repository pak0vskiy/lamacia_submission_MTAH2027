# 2026-11-02 · Data anatomy: what each input is and how it helps the H/A/D label

Author: Maxim. Method: official docs + starter loaders + aggregate-only probes (`analysis/00_schema_probe.py`, `analysis/01_atd_aggregates.py`). No raw rows were printed.

## TL;DR for the team

1. **The ATD has x, y, speed and nothing else.** No ball height, no ball alive/dead flag, no owning
  team. The GT is derived from those flags in the *corrected* tracking, so we must infer both
   from geometry.
2. **The ball is not flickering, it is gone for long blocks.** 150 gaps on graded frames, median
  **10.7 s**, p90 34 s, p99 105 s. **99.7 % of missing-ball frames are in gaps > 5 s.**
   Interpolation will not fill this. Ball-free inference (team shape) or ball re-detection from the
   video is the only way to cover ~41 % of graded frames.
3. **Speed unit is m/s**, despite the XML saying `kph` (player speed p99 = 6.4, p99.9 = 8.1).
4. **The ATD ball speed channel is near-useless**: present on only 8.7 % of frames that have a ball
  position, and its values are implausibly low for a ball (p99.9 = 6.8 m/s). Compute our own ball
   velocity from positions.
5. **The ball is outside the 0–1 pitch box on 4.2 %** of graded frames where it is seen, a free
  dead-ball cue (out of play / throw-in / corner).
6. Metadata gives **2 goalkeeper slots** (`position_type`) and **attack direction per team for the
  1st half**. Useful for goal kicks, GK-held ball, and "which way is forward".
7. Both `.mp4` files are present locally.



## The inputs, one by one



### Tactical video (`BAR-ATH_tactical.mp4`, 5.2 GB, 199,319 frames, 25 fps, 1080p)

- A fixed wide "tactical" camera that sees the whole pitch. **Row *i* of the submission = frame *i*.**
- Role for us: raw source. Only needed if we run our own detector (ball re-detection to fill the
long gaps, or detecting stoppage visuals). Everything else is derived from it already.



### ATD: Automatic Tracking Data (`_atd.xml` + `_atd.txt`)

What it is: Metrica's computer-vision output, **uncorrected**. One line per video frame:
`frame_id : slot1;slot2;…;slot46 : ball`, each group `x,y,speed`.

- `frame_id` = video frame + 1.
- Coordinates normalised 0–1 (×105.3 / ×68.0 for metres). Speed in m/s.
- 46 player slots (24 for team 0 = home/Barcelona, 22 for team 1 = Athletic). A slot is a
tracklet, not a person; it appears, vanishes, swaps. Team membership per slot is reliable
(metadata). ~10.2 objects per team per graded frame (p5 = 6–7).
- The XML adds: team of each slot, 2 goalkeeper slots, attack direction (1st half), field size.

How it helps classification:


| signal                                                                          | from           | gives evidence for                                                   |
| ------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------- |
| ball ↔ nearest H / nearest A distance & margin                                  | ball + players | who is on the ball (H vs A), when the ball is seen                   |
| ball velocity / direction changes (our own, from positions)                     | ball           | touches, passes, interceptions → candidate possession changes        |
| ball outside 0–1, ball static on a line/spot                                    | ball           | **D**                                                                |
| all players slow down, cluster for a set piece                                  | players        | **D** (stoppage)                                                     |
| team centroids, which team moves forward, compactness, who is "behind the ball" | players only   | **H vs A without the ball**, which we need on ~41 % of graded frames |
| GK slot near ball                                                               | ball + GK slot | goal kicks, keeper holding the ball                                  |




### Homography (`_homography.csv`)

- A 3×3 matrix per frame: normalised pitch (0–1) → image pixels. Tactical: 100 % valid, median
reprojection error 0.24 px.
- **It adds no possession information by itself.** It's a bridge between tracking and video:
  - debug: project players/ball onto a frame to see what fooled the model (locally, not in chat);
  - if we detect the ball in pixels ourselves, `unproject` turns it into pitch coordinates
  (ground-plane only: a ball in the air will be mislocated).
- ROI of improving it: ~zero.



### Smart Tagging (`_smart_tagging.xml`)

- 638 automatic intervals (phases + set pieces), seconds from video start → `round(s*25)`.
- Helps as a **prior**, not an answer:
  - set pieces (59) mark restarts = end of a D stretch, and usually who restarts (not for
  throw-ins: `Team = N/A`);
  - attacking phases name the team **in** possession, "Defending in …" names the team **out** of it
  (direction to be verified);
  - attacking/defensive transitions are change-of-possession anchors.
- Phases ≠ possessions; auto-generated, has FP/FN.



### Broadcast video + its ATD + homography

- The TV feed. 164,488 frames, **not synchronised** to tactical, cuts/replays, homography valid on
65.8 %. Its ATD has the same 3-channel format, 51 slots, ball NaN 49.5 %.
- Value: possibly better ball visibility in close-ups, but requires building a broadcast→tactical
time alignment. Low priority, last resort.



### Public GT (`evaluation/gt/…_public.csv`)

- First 5 min of play, 17 possessions. **Sanity only**: alignment, file format, label semantics.
Never tune or select on it.



## What this means for priorities (proposal, to discuss)

- **The ball-free 41 % is the central problem.** With gaps of ~11 s median, a nearest-player
method has nothing to say for whole possessions. Two parallel tracks:
  1. *Team-shape classifier* (players only → H/A/D), trained on other games with labels derived
    from events, ball deliberately hidden to mimic our gap pattern (long blocks, not random drops).
  2. *Ball re-detection on the tactical video* (we have the mp4 + homography), to shrink the gaps.
- **D detection** deserves its own module: ball outside box, stoppage motion signature, set-piece
tags, halftime/kick-off structure.
- **Decoding** (HSMM / state machine with switch costs) on top, since spellF1 punishes flicker.
- Noise model for external data must reproduce **long ball blackouts**, not i.i.d. drop-out.



## Open questions

- Why are ball blackouts so long? Correlated with ball in the air / crowded areas / camera? (aggregate
check: speed of players and team spread at gap start vs end.)
- How many gaps contain a possession change? (Can't use GT beyond 5 min; can use Smart Tagging
transitions as a proxy.)

