# 2026-11-09 · Learned team logic: P(home) per frame + switch-cost Viterbi beats VC by +0.16

Author: Maxim. Code: `possession/learned.py`. Evaluation: `analysis/14_learned_team_cv.py` (held-out
IDSSE, leave-one-match-out) and `analysis/15_learned_on_atd.py` (BAR-ATH against IDSSE, no labels).
Submission: `scripts/make_submission.py` (default `--team learned`). Aggregates only. Follows
[2026-11-09_error-anatomy.md](2026-11-09_error-anatomy.md).

## TL;DR for the team

1. **Held-out spellF1 goes from 0.441 to 0.601 (+0.160), better in 7/7 matches** (min +0.104).
   This uses our D model, calibrated blackouts and a smoothed ball, exactly the VC setup. With
   perfect D it goes from 0.484 to 0.667.
2. **What it is:**
   - A gradient-boosted classifier gives P(home has the ball) for every live frame. Its inputs
     are positions only, looking both backwards and forwards in time:
     - nearest-player distances now, before and after;
     - VC's zone runs and gains;
     - which way each team and the ball are moving relative to that team's attack;
     - team shape;
     - how long the ball has been missing.
   - It's trained on live frames of degraded IDSSE (other games), with every frame duplicated
     with the teams swapped. Its output is exactly team-symmetric.
   - Inside each predicted live stretch, a 2-state Viterbi picks the team sequence, paying a
     fixed cost per switch.
3. **Switch cost 6 was picked in 7/7 folds** (it's a plateau: 4–8 all score 0.598–0.601). Free
   switching (cost 0) gives only 0.459 at a count ratio of 1.47, so the decoder is doing much of
   the work.
4. **It transfers to our match.** On BAR-ATH, with no labels, every aggregate sits on top of
   held-out IDSSE:

   | aggregate (live play, our D) | IDSSE held-out | BAR-ATH |
   |---|---|---|
   | possessions per live minute | 6.25 | 6.19 |
   | possession length q10 / q25 / q50 / q75 / q90 (s) | 1.4 / 2.6 / 5.5 / 11.7 / 23.5 | 1.4 / 2.6 / 5.3 / 12.7 / 24.1 |
   | frames with P(home) in 0.3–0.7 | 11.4 % | 12.3 % |
   | frames with P(home) in 0.1–0.9 | 29.1 % | 31.8 % |
   | agreement with VC where VC has a call | 86.7 % | 84.7 % |
   | home share of H/A time | 52.8 % | 61.0 % |

   Feature quantiles (nearest distance, team speed, forward speed, spread, ball speed) also match
   closely.
5. **Submission:** BAR-ATH graded frames are H 34.0 % · A 21.7 % · D 44.3 % · X 0 %, with 365
   possessions (mean 9.7 s). A from-scratch rebuild gives the same sha256 (`6b58f6e9…`).
   Public 5 min, **sanity only**, not used for any choice: spellF1 0.67 (VC 0.45), 18 possessions
   vs 17 true.

## Numbers (held-out, mean over 7 matches × 3 copies)

| switch cost | 0 | 2 | 4 | **6** | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| spellF1, our D | 0.459 | 0.592 | 0.598 | **0.601** | 0.600 | 0.594 | 0.584 | 0.568 | 0.550 |
| count ratio | 1.47 | 0.86 | 0.81 | 0.78 | 0.75 | 0.71 | 0.68 | 0.64 | 0.61 |

| team logic | D | spellF1 (std) | count ratio |
|---|---|---|---|
| VC TUNED | model | 0.441 (0.018) | 0.81 |
| **learned\*** | model | **0.601** (0.033) | 0.78 |
| VC TUNED | oracle | 0.484 (0.019) | 0.81 |
| learned\* | oracle | 0.667 (0.033) | 0.75 |

Per-frame H/A accuracy on live frames (argmax, held-out): 91.6 % with the ball visible, 81.1 %
with it hidden.

## Caveats

- **The DFL convention is still the target.** It counts many sub-2 s possessions, and Metrica's
  "controlled possession" probably counts fewer. A higher switch cost would suit a stricter
  convention. Checking this on Metrica's sample games (improvement #4) is now the most valuable
  next step, and the switch cost is the parameter it would move.
- **Choosing the switch cost on the same 7 matches' held-out scores** (as for VC\*) adds mild
  optimism. The curve is flat from 4 to 8, so it's small.
- The 0.3 points between this and DFL's own owner (0.90) are still team logic. The count ratio of
  0.78 means short possessions are still merged.
- **Features use future frames.** That's fine offline (the whole match is given), but it means
  this isn't a live model.
- The IDSSE loader still maps every non-home owner to away, so the target labels inherit that.
