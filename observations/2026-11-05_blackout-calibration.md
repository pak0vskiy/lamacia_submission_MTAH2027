# 2026-11-05 · Ball blackouts on IDSSE fitted to BAR-ATH; D and VC re-scored

Author: Maxim. Method:
- `analysis/12_blackout_calibration.py` measures blackouts label-free on both sides;
- `dead_ball/degrade.py: calibrated_mask` simulates them;
- `analysis/06_dead_ball_cv.py` and `analysis/09_team_possession_cv.py` re-score (now default
  `--blackout calibrated`);
- `scripts/make_submission.py` builds the submission in one command.

Aggregates only. On BAR-ATH, live / dead come from our own dead-ball model's calls. No labels,
public or private.

## TL;DR for the team

1. **Our validation hid far too much live play.** On BAR-ATH the ball is missing in **10.6 % of
   live frames** and 76.7 % of dead frames. The old IDSSE degradation hid 31 % / 54 % at c = 0.5
   and 23 % / 65 % at c = 0.9, in live gaps 3× too long, placed at random rather than at duels.
2. **The shape of BAR-ATH's blackouts:**
   - 77 % of stoppages (≥ 2 s) are covered by a gap. The gap starts near the stoppage start
     (median +0.06 s; q10/q90 −5.5 / +12.9 s) and **ends just before the restart** (median
     −0.8 s), so the restart touch is usually visible.
   - Live-only gaps come at 0.64 per live minute. About 1/3 are 1–3 frame blips, the rest
     5–11 s.
   - A live gap is **1.9× more likely after a duel** (a player of each team within 3 m).
3. **The new `calibrated` degradation reproduces it** (measured the same way, D = held-out fold
   model):

   | | BAR-ATH | IDSSE calibrated | IDSSE old c = 0.5 |
   |---|---|---|---|
   | ball missing, live / dead | 10.6 % / 76.7 % | 12.3 % / 70.7 % | 31.4 % / 53.5 % |
   | live gaps per live min | 0.73 | 0.72 | 0.86 |
   | live gap q50 / q90 (s) | 5.5 / 9.7 | 5.7 / 12.8 | 14.2 / 41.6 |
   | stoppages covered by a gap | 77 % | 77 % | 74 % |
   | duel ratio before a live gap | 1.67 | 1.69 | 0.78 |

4. **Re-scored on calibrated copies** (held-out IDSSE, mean over 7 matches):

   | | spellF1 | count ratio |
   |---|---|---|
   | D model alone, DFL teams (setting A), D retrained on calibrated | **0.900** (std 0.011) | 1.03 |
   | same, old D model | 0.886 | 1.04 |
   | **VC\* (zone 2.0 m), new D** | **0.438** | 0.81 |
   | VC\*, old D | 0.431 | 0.80 |
   | VC\*, oracle D | 0.480 | 0.80 |
   | geometry two-stage, new D | 0.347 | 1.37 |
   | v1, new D | 0.347 | 0.44 |

   - **The honest VC expectation rises from 0.374 to 0.438.**
   - Under calibrated blackouts the three "coincidence" columns are just independent test
     copies. They agree within ±0.005, which is the seed noise.
   - VC's zone re-picked 2.0 m in 7/7 folds (2.5 m: 0.433). It's a plateau, and `TUNED` is now
     2.0 m.
5. **The team logic is still the gap:** 0.44 against 0.90 with DFL's owner on the same D. A
   perfect D would add only ~0.04.

## Submission (`uv run python scripts/make_submission.py`)

- D model fitted on all 7 IDSSE matches (calibrated blackouts), with decoder from inner CV
  (enter 0.7 / exit 0.3 / min dead 3 s), then VC `TUNED`. Retraining from scratch reproduces the
  sha256.
- BAR-ATH graded frames: H 34.0 % · A 20.8 % · D 44.3 % · X 0.9 %. 444 possessions in the
  validator's count (X as a gap), mean 7.8 s.
- Public 5 min, **sanity only** (not used for any choice): spellF1 0.45, macroF1 0.82,
  27 possessions vs 17 true.

## Caveats

- **Live / dead on BAR-ATH is our D model's call, not truth.** We measure IDSSE through its own
  held-out D model too, so the comparison is like for like, but the targets inherit D's errors.
  The D model uses time-since-ball-seen, so its "dead" calls correlate with missing ball by
  construction on both sides.
- Dead missing is still 6 points low (70.7 % vs 76.7 %). BAR-ATH stoppages can hold more than one
  gap, and we simulate one.
- The calibration measured 37 live-only gaps and 119 stoppages on BAR-ATH. The quantiles of the
  live-gap length are rough.
- `CAL_P_COVER` (0.95) and `CAL_DUEL_BETA` (1.5) were set so the *measured* coverage and duel
  ratio match. Both are fitted to tracking aggregates, not to any outcome.
- **The 2nd-half pre-kickoff stretch** (graded, all D per the FAQ) comes out only 58 % D. It
  needs a kickoff detector (ball on the centre spot, then moving, inside Smart Tagging's
  kickoff window), not a frame constant.
- `analysis/10_vc_on_atd.py` still uses the old fold models and c = 0.5 copies. It needs a refresh
  before it's quoted again.
