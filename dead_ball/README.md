# Dead-ball model (`dead_ball/`)

This is a tracking-only classifier for the **D** label: whether the ball is out of play (dead) at each
frame. In the possession pipeline it **splits possessions**. spellF1 only scores H/A spells, but
a missed stoppage merges two possessions, and a false short stoppage cuts one in two.

It is trained on **IDSSE** (7 Bundesliga matches from DFL, CC BY 4.0, with a per-frame alive/dead
flag), and that data is degraded to look like our ATD. No BAR-ATH labels are used anywhere.

Background reading, in order:
- Design: [`docs/dead-ball-classifier-design.md`](../docs/dead-ball-classifier-design.md)
- Results: [`observations/2026-11-04_dead-ball-classifier.md`](../observations/2026-11-04_dead-ball-classifier.md)
- Transfer to ATD: [`observations/2026-11-04_dead-ball-transfer.md`](../observations/2026-11-04_dead-ball-transfer.md)
- Label convention: [`observations/2026-11-04_idsse-all-matches.md`](../observations/2026-11-04_idsse-all-matches.md)

## TL;DR

- **Model:** scikit-learn `HistGradientBoostingClassifier` on per-frame windowed features, plus a
  hysteresis decoder that turns P(dead) into stoppages.
- **Chosen config** (the defaults):
  - **without the ball-speed feature** (`DEFAULT_DROP = ("ball_spd_1",)`);
  - trained on **"mixed" dead-ball motion** (`DEFAULT_MOTION = "mixed"`).
- **Held-out IDSSE spellF1** (leave-one-match-out, 7 matches, coincidence c = 0.5):

  | pipeline | no D | v1 as is | **this model** |
  |---|---|---|---|
  | A: DFL's own team labels | 0.768 | – | **0.864** |
  | B: Sam's v1 team logic | – | 0.218 | **0.279** |

- **On BAR-ATH** (the model's own calls, with no labels, so this is a plausibility check only):
  - 101 stoppages per 90 min, median 17.6 s, 45 % dead;
  - IDSSE truth for comparison: 96 per 90 min, 19.6 s, 45 %.
- **The model beat hand rules** by 0.04–0.05 spellF1 in setting A. That is more than the agreed
  0.02 margin, so we use the model, not `rules.py`.

## How it works

### 1. Labels (`idsse.py`)
- The DFL flag is used as is.
- It already follows our convention:
  - the ball goes dead about 0.16 s after it fully leaves the pitch;
  - for fouls, at about the whistle;
  - it comes back alive on the restart touch.
- Loading uses kloppy `sportec.load_tracking(only_alive=False, coordinates="sportec")`, which
  gives metres with the origin at the centre spot.
- `load_match(id)` returns `Match` → `Period`s. Each period holds:
  - `t`, `dead`, `home_owns`;
  - `ball (n,2)`, `home/away (n,P,2)`;
  - goalkeeper slot indices.

  The result is cached in `artifacts/`.

### 2. Making IDSSE look like our ATD (`degrade.py`)
- **Ball blackouts in long blocks:**
  - gap lengths follow our ATD gap quantiles (0.5 / 6.1 / 10.7 / 21.1 / 33.8 s at q10…q90);
  - about 41 % of frames are missing.
- **Blackout–stoppage coincidence `c`:** how strongly gaps prefer dead spells. This is unknown
  on our match, so training draws c ~ U[0, 0.9] and tests report c ∈ {0, 0.5, 0.9}.
- **2 % player drop-outs.** There is no ball z, and every speed is computed from positions.
- **DFL freezes the ball while it's dead, which is a leak.** The model would learn "ball exactly
  still = dead". `dead_ball_path` ("mixed") replaces each frozen spell with:
  - an overshoot along the exit velocity;
  - then 0–2 waypoints to the restart spot, each leg a carry (0.8–2 m/s) or a throw/kick
    (6–15 m/s);
  - then stillness;
  - with 0.5 cm jitter throughout.
- The older `"carry"` mode is a straight line plus 2 cm jitter. It is kept for ablations.

### 3. Features (`features.py: frame_features(p, fps, half_len, half_wid)`)
Per frame, from `x, y` only, over ±1–5 s windows (future context is fine; scoring is offline):

| group | columns |
|---|---|
| player motion | `spd_med_1`, `spd_med_3`, `spd_mean_3`, `spd_med_past_3`, `spd_med_next_3`, `spd_change_3`, `still_share_3`, `n_players` |
| shape | `spread_home`, `spread_away`, `hull_all`, `own_half_share` |
| ball visibility | `ball_vis`, `ball_vis_share_5`, `t_since_ball`, `t_until_ball` (cap 60 s) |
| ball position | `ball_edge` (distance to nearest line, negative outside, forward-filled), `ball_out_last`, `ball_spd_1` (**dropped** in the chosen model) |
| set pieces | `n_near_corner` (6 m), `n_in_box_max`, `n_centre_circle` (9.15 m) |
| goalkeepers | `gk_spd_home`, `gk_spd_away`, `gk_goal_dist_min` |

**What the chosen model relies on** (permutation importance and partial dependence, held out):
- team mean speed over 3 s (P(dead) crosses 0.5 at about 1.45 m/s);
- ball distance to the line (crosses just over the line, about −0.2 m);
- players' speed over the next 3 s;
- time since the ball was last seen.

A depth-4 surrogate tree reproduces 93 % of the model's calls.

### 4. Model (`cv.py: Model`)
`HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05,
max_leaf_nodes=31, early_stopping=False)`.

It is trained on every 5th frame of 2 degraded copies per match.

### 5. Decoder (`pipeline.py: decode_dead(prob, fps, enter, exit, min_dead_s, min_alive_s=1.0)`)
1. Hysteresis: a frame goes dead at `prob ≥ enter` and comes back alive at `prob < exit`.
2. Alive gaps shorter than 1 s are filled.
3. Dead spells shorter than `min_dead_s` are dropped.

The grid is enter {0.5, 0.6, 0.7} × exit {0.3, 0.4, 0.5} × min_dead_s {1, 2, 3}. It is tuned
by inner CV on the training matches only. Typical picks: enter 0.6–0.7, exit 0.3–0.4,
min_dead_s 3.

### 6. Evaluation (`cv.py`, `pipeline.py`, `v1_adapter.py`)
- Leave-one-match-out over the 7 IDSSE matches. The held-out match is used only for the final
  score.
- Scores use the **official** `evaluation/possession_metrics.py` (spellF1, macroF1), plus
  stoppage precision/recall at IoU 0.5.
- **Setting A**: H/A comes from DFL's own ball possession on frames we predict alive. This
  isolates the effect of D.
- **Setting B**: Sam's v1 (`sam_v1.py`, vendored verbatim from `origin/sam_v1`). Our P(dead)
  replaces v1's dead evidence, and 1 − P(dead) replaces its live evidence.
  `v1_as_is` = v1 untouched (the reference).

## Running it

```bash
uv sync
uv run python -m pytest tests/dead_ball -q                       # unit tests
uv run python analysis/06_dead_ball_cv.py                         # 7-fold CV, chosen config
uv run python analysis/07_dead_ball_patterns.py                   # importance, PD, surrogate tree
uv run python analysis/08_atd_vs_idsse_features.py                # BAR-ATH vs IDSSE inputs + model calls
```

Useful flags for `06`:
- `--method rules`: the hand-rule baseline.
- `--drop` with no names: keep every feature.
- `--motion carry`: the old degradation.
- `--matches N`: smoke test on N matches.
- `--bootstrap N`: spellF1 confidence intervals.
- `--tag`: suffix for the output name.

What the scripts write and print:
- Everything is cached or written under `artifacts/dead_ball/`, which is gitignored:
  - prepared copies;
  - per-fold models `fold_<name>_<match>.joblib`;
  - out-of-fold probabilities;
  - per-match CSVs.
- Stdout gets aggregates only.
- The IDSSE XMLs must be in `data/external/idsse/` first. See
  [`docs/idsse-and-dead-ball-model.md`](../docs/idsse-and-dead-ball-model.md) for where to get them. The first run parses
  them with kloppy and takes a while. Later runs use the cache.

**Name of the chosen fold models:** `model-no-ball_spd_1@mixed`.

Using them on new tracking:
1. Build a `Period`.
2. Call `frame_features`.
3. Average `predict_proba` over the 7 fold models.
4. Decode with the most common fold params.

`analysis/08_atd_vs_idsse_features.py` does exactly this for BAR-ATH.

## Files

| file | what |
|---|---|
| `idsse.py` | IDSSE loader → `Match` / `Period`, cached |
| `degrade.py` | ATD-like degradation: blackouts, drop-outs, dead-ball motion |
| `features.py` | per-frame windowed features |
| `cv.py` | data prep, `Model`, `NoDead`, decoder tuning, leave-one-match-out CV, defaults |
| `pipeline.py` | decoder, truth/oracle codes, spellF1 scoring |
| `rules.py` | hand rules distilled from the model (baseline; still uses ball speed) |
| `sam_v1.py`, `v1_adapter.py` | vendored v1 possession pipeline and its IDSSE adapter |

Tests: `tests/dead_ball/` (`conftest.py` provides a synthetic `make_period` factory).

## Caveats and open items

- **The IDSSE numbers are for comparing versions, not for forecasting the leaderboard.**
  Reasons:
  - different league and tracking system;
  - DFL's possession definition ≠ Metrica's "controlled possession".
- **Accuracy on BAR-ATH is unmeasured.** The variant was picked using label-free plausibility
  only: stoppage count, length and dead share.
- **The "mixed" motion overcorrects.**
  - The degraded IDSSE stoppage ball is now too still: slow-play median 0.25 m/s vs 1.87 m/s on
    our ATD.
  - It overshoots too far: −6.5 m vs −2.6 m.
- **Our ball is visible in only 19 % of slow-play frames** (IDSSE 46 %). That sits at the high
  end of the tested coincidence. Results were stable across c.
- **Player drop-out (2 %) is a placeholder**; it has not been measured on ATD.
- **The team half is the bottleneck.** In setting B, v1 predicts only about 35 % as many
  possessions as the truth, with or without our D.
- **Next:**
  - wire the model into `scripts/make_submission.py`;
  - fix v1's team logic.
