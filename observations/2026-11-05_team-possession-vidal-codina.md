# 2026-11-05 · Team logic: Vidal-Codina (2022) possession rule vs v1 vs Sam's geometry decoder

Author: Maxim. Method: `possession/vidal_codina.py` (port of databallpy's
`add_individual_player_possession`), `analysis/09_team_possession_cv.py`. Aggregates only.
Other games only: IDSSE / DFL, 7 matches, degraded like our ATD. Nothing from BAR-ATH is read.

## TL;DR for the team

1. **The published Vidal-Codina rule beats every team logic we had, in all 7 matches.**
   Held-out spellF1 with our dead-ball model as D, at c = 0.5:

   | team logic | spellF1 | macroF1 |
   |---|---|---|
   | **VC tuned** (pz 1.0 m) | **0.523** | 0.830 |
   | VC, paper defaults (pz 1.5 m) | 0.508 | 0.825 |
   | Sam's geometry two-stage | 0.331 | 0.780 |
   | Sam's v1 | 0.279 | 0.720 |
   | *DFL's own owner (ceiling for this D)* | *0.864* | *0.933* |

   The match-by-match margin over geometry is at least 0.135, and over v1 at least 0.155.
2. **The rule:** the closest player within a possession zone gains the ball only if the ball's
   speed jumps by more than 5 m/s between frames, or its direction turns by more than 10°.
   - The team then keeps the ball until the other team gains, through passes in flight and
     ball blackouts.
   - Ownership is reset at every predicted stoppage.
3. **Tuning:** we only shrank the possession zone, from 1.5 to 1.0 m. That choice was made
   leave-one-match-out on the other 6 matches and picked 1.0 m in 7 of 7 folds. Speed 5 m/s was
   picked in 7/7 folds and angle 10° in 6/7 (all c = 0.5).
4. **Team logic is now the bottleneck, not D.** With perfect D (the DFL flag), VC scores 0.567
   (0.582 tuned). So our D model costs about 0.06 and the team logic about 0.3.
5. **Sam's two-stage geometry decoder produces about the right number of possessions** (count
   ratio 1.05) **but puts their boundaries in the wrong place**. Its 0.24 s switch confirmation
   flips team whenever the other team is nearer for a moment.

## Numbers (mean over 7 held-out matches)

| team logic | D | c=0.0 | c=0.5 | c=0.9 | possession count ratio (c=0.5) |
|---|---|---|---|---|---|
| VC defaults | model | 0.461 | 0.508 | 0.541 | 0.79 |
| VC tuned | model | – | 0.523 | – | 0.72 |
| geometry | model | 0.319 | 0.331 | 0.356 | 1.05 |
| v1 | model | 0.258 | 0.279 | 0.309 | 0.37 |
| VC defaults | oracle | 0.524 | 0.567 | 0.595 | 0.80 |
| VC tuned | oracle | – | 0.582 | – | 0.70 |

- The count ratio is predicted possessions / true possessions. DFL has about 437 per match.
- std of spellF1 over matches at c = 0.5 is ≤ 0.05 everywhere.
- Scores rise with c: when ball blackouts fall in stoppages, fewer in-play gains are hidden.

## Caveats

- **VC undercounts possessions (about 0.75×).** Gains hidden by ball blackouts are missed, and
  short possessions merge. That's the next thing to attack, e.g. with player-motion cues during
  blackouts or Sam's learned owner model as a fallback.
- **The DFL BallPossession convention differs from Metrica's "controlled possession"**, so the
  absolute numbers transfer only roughly. The ranking of methods is the robust part.
- **Ball speed matters here, and that's a transfer risk.** VC relies on ball speed changes, and
  our ATD ball moves differently from degraded IDSSE (see
  `2026-11-04_dead-ball-transfer.md`). The jump threshold sits at the top of the grid (5 m/s),
  so it should be checked on BAR-ATH aggregates: gains per minute, without labels.
- **The truth labels count unowned in-play frames as away.** The IDSSE loader maps every
  non-home owner to away, so this applies to all methods equally. We still need to count how
  often it happens.
- Sam's learned owner model (`owner_model.py`) is not in this comparison yet. It needs
  per-fold training to be scored without leakage.
