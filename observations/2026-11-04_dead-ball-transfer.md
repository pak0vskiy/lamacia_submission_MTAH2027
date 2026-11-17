# 2026-11-04 · Dead-ball model: does it transfer to our ATD? Ball speed out, realistic dead ball in

Author: Maxim. Method: `analysis/08_atd_vs_idsse_features.py` (label-free BAR-ATH vs degraded IDSSE),
`analysis/06_dead_ball_cv.py --drop … --motion …` (ablations), `analysis/07_dead_ball_patterns.py`.
Aggregates only. BAR-ATH side uses graded-frame tracking only: **no GT, public or private**.
Follows [2026-11-04_dead-ball-classifier.md](2026-11-04_dead-ball-classifier.md).

## TL;DR for the team

1. **Player signals transfer; the ball doesn't.** During slow play (team mean speed < 1 m/s, a
   label-free stoppage proxy), player-speed and stillness quantiles match BAR-ATH vs IDSSE almost
   exactly. The ball doesn't. Our stoppage ball is either really still or moving fast (median
   1.87 m/s, q90 10.9). The degraded IDSSE ball sat at ~0.9 m/s, an artefact of the 2 cm jitter
   we added to undo DFL's frozen dead ball. Our ATD also tracks the ball further past the line
   (q10 −2.6 m vs −0.3 m), and our ball is visible in only 19 % of slow-play frames (IDSSE 46 %).
2. **The original model chopped up BAR-ATH stoppages**: 111 per 90 min, median 12.0 s, against
   96 / 19.6 s in IDSSE truth. That's what a mis-read moving dead ball looks like. Every fake
   short possession costs spellF1.
3. **Two fixes, tried separately and together:**

   | variant | IDSSE spellF1, A (DFL teams) | B (v1 teams) | BAR-ATH stoppages / 90 min | BAR-ATH median stoppage |
   |---|---|---|---|---|
   | original (straight carry + ball speed) | **0.881** | 0.275 | 111 | 12.0 s |
   | no ball-speed feature | 0.861 | 0.275 | 100 | **18.4 s** |
   | "mixed" dead-ball motion | 0.870 | 0.274 | 100 | 15.0 s |
   | **both (chosen)** | 0.864 | **0.279** | 101 | 17.6 s |
   | *IDSSE truth* | | | *96* | *19.6 s* |

   (Held-out IDSSE, c = 0.5; BAR-ATH columns are the model's own calls, plausibility only.)
4. **Chosen: no ball speed + mixed motion** (now the defaults in `dead_ball/cv.py` and the scripts).
   - Cost: ~0.02 spellF1 on IDSSE in A, none in B.
   - BAR-ATH stoppage count and length move to the IDSSE pattern.
   - Predicted dead share on BAR-ATH 45.4 % (IDSSE truth 44.8 %).
5. **What the chosen model uses:** team mean speed over 3 s (P(dead) crosses 0.5 at 1.45 m/s),
   ball distance to the line (crosses at about −0.2 m, i.e. just over the line), players' speed in
   the next 3 s, time since the ball was last seen. A depth-4 tree reproduces 93 % of its calls.

## The "mixed" dead-ball motion (`dead_ball/degrade.py: dead_ball_path`)

DFL freezes the ball position while dead. We replace each frozen spell as follows:

1. The ball overshoots along its exit velocity for 0.5–2.5 s, decelerating.
2. It then travels through 0–2 random waypoints to the restart spot. Each leg is either a carry
   (0.8–2 m/s) or a throw / kick (6–15 m/s, 40 % of legs).
3. The remaining time is spent still.
4. 0.5 cm tracking jitter is added everywhere.

## Caveats

- **Picking the variant used BAR-ATH tracking aggregates.** These are unlabelled (no GT), so this
  is allowed, but it is a plausibility match, not accuracy. Real accuracy on our match remains
  unmeasured.
- **The mixed motion overcorrects.** The IDSSE stoppage ball is now too still (slow-play median
  0.25 vs 1.87 m/s) and overshoots too far (−6.5 vs −2.6 m). With ball speed out of the model,
  this mostly affects `ball_edge`.
- Our ball visibility in stoppages (19 %) is at or beyond the high end of the coincidence range
  we test (c up to 0.9). Results were stable across c, which is reassuring.
- The hand rules (`dead_ball/rules.py`) still use ball speed. They're not the chosen component.
