# Dead-ball classifier (tracking-only baseline): design

Date: 2026-11-04. Author: Maxim. Status: implemented.

## Intent and why

The competition ranks on **spellF1**: only H and A possessions are scored, each matched one-to-one
against the truth by temporal IoU (thresholds 0.3–0.7). D frames are never scored directly in
spellF1, but D is what **separates and trims possessions**:

- H → D → H is two true possessions; missing the stoppage merges them and loses both.
- Dead spells are long (IDSSE median ~20 s). Gluing one onto a neighbouring possession can push
  its IoU below threshold, even when the neighbour is the other team.
- A false short D inside a possession splits it in two. D precision matters as much as recall.
- Boundary precision of ±1 s barely moves IoU. Getting the right stoppages matters; frame-exact
  edges don't (they only feed the macroF1 tie-break).

So the classifier is a **possession splitter** inside the possession pipeline. We judge it by
the spellF1 of the pipeline it sits in, not by frame accuracy on D.

Our match's ball is missing ~41 % of graded frames in long blocks (median gap 10.7 s), often
around stoppages. So the classifier must mostly infer stoppages from player behaviour, not from
the ball.

## Goals

1. A **tracking-only** D classifier (no video, no audio): the baseline later additions get
   measured against.
2. Learn it on IDSSE (7 DFL matches, per-frame alive/dead flag), with the data degraded to look
   like our ATD.
3. **Learn first, then distil to rules.** Train a model, read out what it relies on, write
   hand rules from that, and compare the two head to head.
4. Score with the **official spellF1** (`evaluation/possession_metrics.py`) on held-out IDSSE
   matches, in two pipeline settings (below).

## Non-goals

- Video, audio, whistle detection, ball re-detection.
- Improving the H/A (team) logic. It's an input here, not a deliverable.
- Any use of BAR-ATH labels, including the public 5 min (sanity only, never for selection).
- Producing the competition submission. That's a later step once a D component is chosen.

## Anti-goals (failure even if it "works")

- A model that wins on IDSSE by using things our ATD doesn't have: ball z, a continuously present
  ball, DFL-specific data quirks, ATD's broken speed channel semantics.
- Tuning hand rules on the same match they're scored on.
- Anything frame-specific or game-specific to BAR-ATH.

## Constraints

- Repo rules: aggregates only to stdout; per-frame arrays in `artifacts/`; run via
  `uv run python …`; fixed seeds, pinned deps; log decisions in `docs/decisions.md`; findings in
  `observations/`.
- External data (IDSSE, CC BY 4.0) must be documented in `REPORT.md`.
- Labels: the DFL flag **as-is, no shift**. `observations/2026-11-04_idsse-all-matches.md`
  shows it already follows our convention: dead 0.16 s after the ball leaves the pitch, around
  the whistle for fouls, alive on the restart touch.
- IDSSE loading: `only_alive=False`, `coordinates="sportec"` (metres, centre origin). Reuse the
  cache in `artifacts/idsse_frames_<match>.npz` from `analysis/05_idsse_all_matches.py`. Extend
  it with player positions; the current cache holds ball, flag, owner and player count only.

## Decided (by the user)

1. **Tracking only** for the baseline.
2. **Learned model first, then analyse its patterns, then hand-picked rules informed by them.**
3. **Deciding rule:** if the hand rules come within **0.02 spellF1** of the model (mean over
   held-out matches, same folds for both), use the rules. Otherwise use the model. Check the
   bootstrap intervals (`bootstrap_spell_f1`) before calling a difference real.
4. **Two evaluation settings, both reported:**
   - **A, DFL-oracle teams.** H/A comes from DFL's own `BallPossession` on frames we predict
     alive. This isolates the D classifier's effect on spellF1.
   - **B, Sam's v1 team logic.** The classifier's output replaces v1's dead-ball evidence inside
     `lamacia_possession_v1.py` (branch `origin/sam_v1`). v1's own owner evidence and decoder
     stay as they are. Also run v1 unmodified as the reference to beat.

## How it works

### 1. Degradation to look like our ATD (applied to every IDSSE frame, train and test)

- **Ball blackouts in long blocks**, durations drawn to match our gap distribution (q10/25/50/75/90
  = 0.5 / 6.1 / 10.7 / 21.1 / 33.8 s), with a total missing share of ~41 %.
- **Blackout–stoppage coincidence is unknown on our match** (it would need GT). The blackout
  placement has a parameter for how strongly gaps prefer dead spells. It's drawn per training
  copy from a range covering "independent of D" to "~80 % of missing frames inside stoppages".
  (`observations/2026-11-02_ball-gaps.md` gives ~80 % of missing frames in gaps with a stoppage
  signature: an aggregate cue, not a label.) Each test fold is scored at several fixed
  coincidence levels and the spread is reported. The choice has to be robust across the range.
- **No ball z.** Ball velocity comes from positions only (as for our ATD).
- **Player dropout / fragmentation**: random short player gaps. Rate is my call (see below)
  until someone measures ATD player-slot coverage in aggregate.
- Player speeds are computed from positions, never from a provider speed channel.

### 2. Signals (starting list, all over trailing + centred windows of ~1–5 s)

- team-wide speed (median, mean) and its change; share of players near-stationary
- player spread and compactness per team (convex-hull area, centroid distances)
- ball visible flag, time since last seen, time until next seen (offline, so future context is
  allowed)
- ball distance to nearest line, ball left pitch (last seen outside), ball stillness
- set-piece shape: players clustered near a corner flag, penalty area, centre circle (kick-off),
  a wall-like line
- goalkeeper position and speed relative to their box (goal kicks)

The builder may add or drop signals. Every one has to be computable from ATD's `x, y` per object.

### 3. Model

- Gradient-boosted trees on per-frame windowed features, binary dead/alive. Default:
  scikit-learn `HistGradientBoostingClassifier` with a fixed seed (my call: no new heavy deps,
  deterministic).
- **Leave-one-match-out**: 7 folds. All selection (features, hyperparameters, decoder
  thresholds) happens inside the 6 training matches; the held-out match is touched only for the
  final score.

### 4. From scores to stoppages (decoder)

- Hysteresis on the dead probability plus a **minimum dead-spell length** (~2 s; DFL has 3
  spells < 1 s in 713) and a minimum alive gap between stoppages. Thresholds are picked on the
  training folds by spellF1.
- Same decoder for the model's and the rules' output, so the comparison is about the D signal.

### 5. Pattern read-out → hand rules

- Permutation importance per fold. Partial dependence for the top signals, to read off threshold
  locations. A shallow surrogate tree (depth ≤ 4) fit to the model's predictions.
- Hand rules written from that read-out, with thresholds fit on the training folds only. Same
  decoder. Same 7-fold scoring.

### 6. Pipeline settings for scoring

- Truth codes per IDSSE frame: D where the flag is dead; else H/A from `BallPossession`. Valid
  runs = the two periods.
- **A, oracle**: predicted D → D; predicted alive → DFL owner at that frame. Where the truth is
  dead (owner unreliable), use the owner of the last alive frame (my call).
- **B, Sam's v1**: build v1's feature frame (`median_speed, n_speed, ball_x_m, ball_y_m,
  ball_visible, d_home_m, d_away_m, edge_m`) from degraded IDSSE with an adapter. Convert
  coordinates to v1's 0…length / 0…width metres and the half boundaries to IDSSE periods. Run
  v1's `evidence()` for owner/live, substitute our dead score for its `dead`, then run its
  `decode()`. Also run v1 untouched (reference).
- Report per fold and pooled: spellF1 (mean over IoUs), macroF1, D-spell precision/recall,
  predicted vs true possession counts.

## Deliverables

- Code in `analysis/` (or a `dead_ball/` package if it grows), runnable by one command per stage,
  aggregates only to stdout.
- `observations/2026-11-XX_dead-ball-classifier.md`: settings A and B tables, the pattern
  read-out, the rules, the 0.02 decision and which branch won.
- Entries in `docs/decisions.md` and an IDSSE section in `REPORT.md`.

## Left to the builder

- Exact window lengths, feature formulas, hyperparameter grid.
- Player-dropout rate (placeholder; measure ATD slot coverage in aggregate if cheap).
- How to import v1 from `origin/sam_v1` (vendor a copy with attribution, or check out the file).
  It must not change v1's behaviour.
- Code layout and caching.

## Open risks

- **Domain gap** (Bundesliga, DFL tracking vs La Liga / Metrica ATD). The IDSSE spellF1 numbers
  are for comparing versions, not a forecast of our leaderboard score.
- **DFL `BallPossession` ≠ Metrica "controlled possession"**: setting A's absolute numbers inherit
  that difference.
- v1 thresholds were designed for ATD. Setting B may under-represent v1 on IDSSE; that's another
  reason to read A and B together.
