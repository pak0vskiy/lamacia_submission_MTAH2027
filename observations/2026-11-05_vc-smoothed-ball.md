# 2026-11-05 · Vidal-Codina on a smoothed ball: re-tuned for our ATD

Author: Maxim. Method:
- `analysis/11_ball_smoothing.py` measures how smooth our ball is, label-free;
- `dead_ball/degrade.py: smooth_ball` applies that smoothing to IDSSE;
- `analysis/09_team_possession_cv.py --grid wide|refine` scores the rule leave-one-match-out on IDSSE;
- `analysis/10_vc_on_atd.py` checks BAR-ATH against IDSSE, label-free.

Aggregates only. No BAR-ATH labels used. Follows
[2026-11-05_vidal-codina-rules.md](2026-11-05_vidal-codina-rules.md).

## TL;DR for the team

1. **Our ATD ball is smoothed.** A Gaussian with σ = 0.25 s on the DFL ball reproduces our
   ball's frame-to-frame speed changes in open play: q90/q99 0.61/1.26 m/s vs ATD 0.68/1.19. The
   raw DFL ball gives 1.34/12.0. The open-play proxy is players' mean speed > 1.5 m/s, with no
   labels.
2. **Once IDSSE's ball is smoothed the same way, the earlier tuning no longer holds:**
   - with a 1.0 m zone the rule drops from spellF1 0.523 to 0.31;
   - the speed trigger never fires (0 % of gains);
   - the smoothed ball track sits further from the feet.
3. **Re-tuned leave-one-match-out on smoothed IDSSE: zone 2.5 m, angle 10°, single-frame
   steps.**
   - The folds picked 2.5 m in 4 of 7 and 2.0 m in 3 of 7.
   - The speed threshold doesn't matter on a smoothed ball.
   - The window-based speed trigger we built for this (`window_frames > 1`) **did not help**:
     at the best zone sizes it adds false gains (count ratio > 1).
4. **Expected spellF1 drops, but VC still wins in all 7 matches.** Held-out, with our D model,
   c = 0.5, smoothed ball:

   | team logic | spellF1 | count ratio |
   |---|---|---|
   | **VC tuned (2.5 m, 10°)** | **0.374** | 0.78 |
   | VC paper defaults (1.5 m) | 0.360 | 0.54 |
   | Sam's geometry two-stage | 0.298 | 1.09 |
   | v1 | 0.279 | 0.37 |
   | *DFL owner (ceiling for this D)* | *0.864* | *1.05* |

   - VC beats geometry by ≥ 0.015 and v1 by ≥ 0.06 in every match.
   - **The earlier 0.523 assumed a raw DFL ball. 0.374 is the honest expectation for our data.**
5. **Our match now behaves like IDSSE** (rule's own calls, no labels):

   | aggregate (live play) | IDSSE smoothed, c = 0.5 | BAR-ATH |
   |---|---|---|
   | frame-to-frame speed change q50 / q90 / q99 (m/s) | 0.15 / 0.62 / 1.27 | 0.20 / 0.67 / 1.22 |
   | ball within 2.5 m of a player, of visible live frames | 63.5 % | 60.6 % |
   | gain trigger: direction only | 100 % | 100 % |
   | gains per live minute | 12.6 | 17.4 |
   | possessions per 90 min | 301 predicted (412 true) | 411 |
   | possession length q10 / q25 / q50 / q75 / q90 (s) | 0.6 / 1.6 / 4.2 / 11.3 / 24.9 (true 0.8 / 1.7 / 3.7 / 8.5 / 18.6) | 0.8 / 1.7 / 4.1 / 9.3 / 17.3 |
   | live frames left X | 8.9 % | 1.3 % |

   - BAR-ATH finds more gains than IDSSE because its ball is visible in 89 % of live frames,
     against 69 % on degraded IDSSE.
   - Its count and length distribution land close to IDSSE truth. That's encouraging, but it is
     plausibility, not accuracy.

## Updated parameters (replace the values in the rules note)

| rule | parameter | was (raw DFL ball) | **now (smoothed, our ATD)** |
|---|---|---|---|
| P1 possession zone | `pz_radius` | 1.0 m | **2.5 m** |
| P3a speed jump | `bv_threshold` | 5 m/s | 5 m/s (never fires on a smoothed ball) |
| P3b direction change | `ba_threshold` | 10° | **10°** |
| window | `window_frames` | 1 | **1** (window trigger tested, rejected) |

The code constant is `possession.vidal_codina.TUNED`.

## Caveats

- **This is a second selection on the same 7 matches.** Picking the "refine" grid after seeing
  where "wide" peaked adds mild optimism. Each fold's choice is still leave-one-match-out, and
  the optimum is a flat plateau (2.0 to 2.5 m, all 0.370 to 0.374), so it should be small.
- **The smoothing σ was matched to BAR-ATH tracking aggregates** (no labels): one number,
  matched on the ball's motion statistics. It isn't tuned on any outcome.
- **The D model was trained on the unsmoothed ball.** Its features hardly use ball speed (it was
  dropped), but retraining D on smoothed IDSSE is a cheap consistency check worth doing.
- **The ceiling of this rule family on a smoothed ball is about 0.37.** The gap to the DFL
  owner (0.86) is mostly in team logic. Next candidates: a learned owner model trained on
  smoothed IDSSE (Sam's `owner_model.py`, retrained per fold), and gains from player-motion cues
  when the ball is unseen.
