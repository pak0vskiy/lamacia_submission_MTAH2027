# 2026-11-05 · Team possession rules (from Vidal-Codina 2022), and how they behave on our match

Author: Maxim. Code: `possession/vidal_codina.py`. Evaluation: `analysis/09_team_possession_cv.py`
(IDSSE, held-out spellF1), `analysis/10_vc_on_atd.py` (BAR-ATH vs IDSSE, label-free). Aggregates
only. **No BAR-ATH labels used, public or private.** Follows
[2026-11-05_team-possession-vidal-codina.md](2026-11-05_team-possession-vidal-codina.md).

## TL;DR for the team

1. **The rules (below) are our team-logic baseline.** On held-out IDSSE they score spellF1 0.523
   on our D, against 0.331 for Sam's geometry decoder and 0.279 for v1.
2. **On our match, half of the rule is dead: the ball-speed trigger never fires.**
   - The ATD ball changes speed smoothly. Frame-to-frame speed changes in live play have
     q99 = 1.2 m/s on BAR-ATH, against 12.4 m/s on degraded IDSSE.
   - Only 0.01 % of live frames exceed the 5 m/s threshold, against 2.8 % on IDSSE.
   - So 100 % of BAR-ATH gains come from the direction trigger. On IDSSE, 53 % have a speed
     trigger.

   This looks like provider smoothing of the ball track. It's a property of the data, not of
   the game.
3. **The direction trigger compensates partly.**
   - Gains: 11.2 per live minute on BAR-ATH, against 13.8 on IDSSE.
   - Possessions: 248 per 90 min on BAR-ATH, against 294 predicted and 412 true on IDSSE.
   - Median possession: 8.2 s on BAR-ATH, against 4.7 s predicted and 3.7 s true on IDSSE.

   So on our match we probably miss even more possessions than on IDSSE (where we already find
   about 0.75 of them), and merged possessions cost spellF1.
4. **Sanity signs are fine.**
   - Only 2 % of live time is left unlabelled (X).
   - The home team (Barcelona) has 61 % of H/A time, which is plausible.
   - Our ball is visible in 89 % of *live* frames, against 69 % on degraded IDSSE. Our blackouts
     sit mostly in stoppages, which helps here.
5. **Fix (next):**
   - Make the IDSSE degradation smooth the ball the way ATD does, so training and tuning see
     the same ball.
   - Replace the frame-to-frame speed jump with a speed change measured across the possession
     zone (entry vs exit), which survives smoothing.
   - Re-tune both leave-one-match-out on IDSSE.

## The rules

> **Updated parameters:** on a ball smoothed like our ATD the zone is **2.5 m** and the speed
> trigger never fires. See [2026-11-05_vc-smoothed-ball.md](2026-11-05_vc-smoothed-ball.md). The
> values in the tables below are the raw-DFL-ball tuning.

Ported from databallpy 0.8.1 (`add_individual_player_possession`, MIT), which implements
Vidal-Codina, Evans, El Fakir & Billingham, *Automatic event detection in football using tracking
data*, Sports Engineering 25 (2022). Coordinates are metres; 25 fps.

**Player level: who has the ball**

| # | rule | parameter (ours / paper default) |
|---|---|---|
| P1 | **Possession zone.** In each frame, the player closest to the ball is a *candidate* if within `pz_radius`. If the ball is missing, nobody is. | `pz_radius` **1.0 m** / 1.5 m |
| P2 | **Candidate run.** Consecutive frames with the same candidate form one run: the ball in that player's zone. | – |
| P3a | **Gain by speed.** The run is a gain if the ball's speed changes by more than `bv_threshold` between any two consecutive frames inside the run. The kick or the stop shows as a speed jump. | `bv_threshold` **5 m/s** / 5 m/s |
| P3b | **Gain by direction.** Or, the run is a gain if the ball's direction turns by more than `ba_threshold`. The angle is between the step entering the zone (frame before the run → first frame) and the step leaving it (last frame → frame after). | `ba_threshold` **10°** / 10° |
| P3c | **Minimum run length** (frames) for a gain. | `min_frames_pz` 0 / 0 |
| P4 | **Possession lasts while the ball is in the zone.** It ends when the ball leaves the zone (the run ends). | – |
| P5 | **Same player again extends.** If the next accepted candidate is the same player, the possession is extended over the gap, with no new gain needed (dribbling in and out of the zone). | – |
| P6 | **No gain, no possession.** A run that fails P3 (the ball rolls past a player unchanged) leaves possession with nobody. | – |

**Team level: H / A / D / X per frame (our addition, for spellF1)**

| # | rule |
|---|---|
| T1 | **Dead first.** Frames our dead-ball model marks dead are D. Team rules never change D. |
| T2 | **Hold until the other team gains.** Within a live interval, the team of the last gain keeps the ball through passes in flight, loose balls, ball blackouts, and gains by its own players. |
| T3 | **Back-fill the restart.** Live frames before the first gain of an interval get that gain's team. This covers the restart touch, which usually happens just before the ball is tracked into a zone. |
| T4 | **Reset at every stoppage and half.** No team is carried across D. |
| T5 | **Unknown stays unknown.** A live interval with no gain at all is X. We don't guess. |

**How the parameters were chosen.** Grid: pz {0.5, 0.75, 1.0, 1.5, 2.0} × speed {2, 3.5, 5} ×
angle {10, 20, 45}. Each held-out IDSSE match uses the best mean spellF1 on the other 6 (c = 0.5).
- pz 1.0 m was chosen in 7/7 folds.
- Speed 5 m/s was chosen in 7/7 folds. That's the top of the grid, and moot on our match (point 2).
- Angle 10° was chosen in 6/7 folds.

## Check on our match (label-free, `analysis/10_vc_on_atd.py`)

| aggregate (live play) | IDSSE degraded, c = 0.5 | BAR-ATH |
|---|---|---|
| live share of frames | 56.7 % | 54.6 % |
| ball visible, of live frames | 68.6 % | 89.1 % |
| ball within 1.0 m of a player, of visible live frames | 38.0 % | 28.6 % |
| frame-to-frame speed change q50 / q90 / q99 (m/s) | 0.41 / 1.38 / 12.42 | 0.20 / 0.67 / **1.22** |
| frames with speed change > 5 m/s | 2.82 % | **0.01 %** |
| gains per live minute | 13.8 | 11.2 |
| gain trigger: speed only / direction only / both | 6 / 47 / 47 % | **0 / 100 / 0 %** |
| possessions per 90 min (predicted) | 294 (truth 412) | 248 |
| possession length q10 / q25 / q50 / q75 / q90 (s) | 0.7 / 1.8 / 4.7 / 11.8 / 25.2 (truth 0.8 / 1.7 / 3.7 / 8.5 / 18.6) | 1.6 / 3.4 / 8.2 / 17.3 / 26.9 |
| live frames left X | 8.1 % | 2.0 % |
| home share of H/A frames | 53.1 % | 61.1 % |

D on both sides comes from our dead-ball model: the 7-fold mean on BAR-ATH, the match's own fold
on IDSSE. BAR-ATH numbers are the rule's own calls, so this is a plausibility check, not accuracy.

## Caveats

- **Smoothing of the ball track is inferred from aggregates, not documented.** It also affects
  P3b, because direction is measured over single-frame steps. If both triggers are softened,
  the zone may catch slightly different moments.
- **The ball sits in a zone less often on BAR-ATH** (28.6 % vs 38.0 %). This fits smoothing,
  which pulls the ball track away from the feet, or a different player-centre convention. A
  slightly larger zone may suit ATD. That should be tuned only through a better-matched IDSSE
  degradation, never on BAR-ATH.
- **DFL possession ≠ Metrica "controlled possession"**, so the counts compare only roughly.
