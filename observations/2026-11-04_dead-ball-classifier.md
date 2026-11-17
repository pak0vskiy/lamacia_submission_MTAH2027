# 2026-11-04 · Dead-ball classifier (tracking only): model vs hand rules

Author: Maxim. Method: `dead_ball/` package, `analysis/06_dead_ball_cv.py` (CV),
`analysis/07_dead_ball_patterns.py` (read-out). Aggregates only. Other games only: IDSSE / DFL,
7 matches, degraded to look like our ATD. Nothing from BAR-ATH labels is used. Design:
`docs/dead-ball-classifier-design.md`.

## TL;DR for the team

1. **The classifier pays off as a possession splitter.** With DFL's own possession labels for the
   teams (setting A), held-out spellF1 goes from **0.768 with no D → 0.881 with the model**.
   Plugged into Sam's v1 team logic (setting B): **0.218 (v1 as is) → 0.275**. Everything is
   leave-one-match-out over 7 matches, c = 0.5. Stable across blackout–stoppage coincidence
   0 / 0.5 / 0.9.
2. **Model wins the head-to-head.** Hand rules from the model read-out: A 0.836, B 0.269. The paired
   model − rules gap in A is 0.038–0.049 (model better in 7/7 matches), more than the agreed 0.02
   margin. **Decision: use the model.** In B the gap is only 0.006 because v1's team logic
   dominates the error there.
3. **Leak found and fixed: DFL freezes the ball while dead.** Exactly still ball: 100 % of DFL dead
   frames vs 0.8 % of alive frames; our ATD 0.1 % of visible frames. The first model keyed on it
   (A 0.905, now discarded). The degradation now carries the ball from freeze point to restart
   point with 2 cm jitter.
4. **What the model looks at** (permutation importance, held-out): team mean speed over 3 s, ball
   speed, ball distance to the line, players' speed in the next 3 s. P(dead) crosses 0.5 at about
   1.45 m/s team speed, 1.2 m/s ball speed, 0.5 m from the line, 0.5 m/s next-3 s median. A depth-4
   tree on these reproduces 94 % of its calls. Set-piece shapes and goalkeeper features matter
   little.
5. **v1's team logic is the bigger problem on IDSSE.** v1 predicts only ~35 % as many possessions
   as the truth (count ratio 0.35), with or without our D. That's the next thing to fix. Caveat:
   DFL possession ≠ Metrica "controlled possession", and v1 was designed for ATD.

## Numbers (mean over 7 held-out matches; std ≤ 0.04 everywhere)

| setting | method | c=0.0 | c=0.5 | c=0.9 | macroF1 (c=0.5) | D-spell P / R @IoU 0.5 | possession count ratio |
|---|---|---|---|---|---|---|---|
| A (DFL teams) | none (never D) | 0.768 | 0.768 | 0.768 | 0.473 | – | 0.93 |
| A | **model** | **0.882** | **0.881** | **0.887** | 0.951 | 0.84 / 0.86 | 1.06 |
| A | rules | 0.833 | 0.836 | 0.850 | 0.927 | 0.73 / 0.83 | 1.07 |
| B (v1 teams) | v1 as is | 0.201 | 0.218 | 0.256 | 0.634 | 0.36 / 0.21 | 0.35 |
| B | v1 + model D | 0.260 | 0.275 | 0.301 | 0.732 | 0.62 / 0.45 | 0.36 |
| B | v1 + rules D | 0.254 | 0.269 | 0.295 | 0.718 | 0.58 / 0.44 | 0.36 |

Columns c = blackout–stoppage coincidence of the held-out copy (unknown for BAR-ATH). Decoder
chosen by inner CV: enter 0.6–0.7, exit 0.3–0.4, minimum dead spell 3 s.

The rules (thresholds fit on training folds):
- R1: players slow, not about to speed up, ball slow or unseen.
- R2: players slow, ball at or over a line.
- R3: ball slow near a line.

See `dead_ball/rules.py`.

## Caveats

- **IDSSE scores are for comparing versions, not a leaderboard forecast**: different league,
  different tracking system, DFL possession convention.
- **The carried-ball fix is a guess** at what our ATD shows during stoppages. Ball speed is still
  the #2 feature. If our ATD's dead-ball ball motion looks different (e.g. erratic, or a new ball
  appearing at the restart spot), the model may transfer worse than A suggests. Next check:
  compare the distribution of `ball_spd_1` on BAR-ATH vs degraded IDSSE (aggregates, no labels).
- Player dropout (2 %) is a placeholder, not measured on ATD.
- Ball blackouts are placed synthetically. Their real relation to stoppages is unknown, which is why
  c is varied. The model is robust across c.

## Next

- Ball-speed distribution check (above), then plug the model into the submission pipeline.
- Fix the team half: v1's possession count is a third of the truth on IDSSE.
