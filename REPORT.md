# LaMacIA: frame-level ball possession for BAR-ATH

Technical report for More than a Hack 2027, Phase 1 (Ball Possession).

**One command reproduces the submitted file:** `uv run python scripts/make_submission.py`
(details in [Reproduction](#8-reproduction)).

---

## 1. Summary

For every graded frame, the submission says whether the ball is dead (**D**), or alive and held by
the home (**H**) or away (**A**) team. Frames outside the graded windows are **X**. The pipeline has
three stages, and every model and threshold in it was fitted and chosen on **other games**:

```
BAR-ATH ATD tracking (x, y per object, 25 fps)
        │
        ├─► 1. Dead-ball model ─────────────► dead / live per frame
        │      gradient-boosted trees on tracking-only features,
        │      trained on 7 IDSSE (DFL) matches degraded to look like our ATD,
        │      hysteresis decoder (enter 0.7 / exit 0.3 / min dead 3 s)
        │
        ├─► 2. Team logic, inside each predicted live stretch ──► H / A per frame
        │      gradient-boosted P(home has the ball) from positions only,
        │      then a 2-state Viterbi with a fixed cost of 6 per team switch
        │
        └─► 3. Opening kickoff ──────────────► D before the half's opening kick
               Law-8 formation detector (positions only), accepted only
               inside Smart Tagging's automatic "Kick Off (Start)" interval
```

| what | value |
|---|---|
| expected spellF1, held-out IDSSE (other games, leave-one-match-out) | **0.601** (previous rule-based team logic: 0.441) |
| graded frames in the submission | H 33.2 % · A 21.7 % · D 45.1 % · 362 possessions |
| public 5-minute window, **sanity only** (never used to choose anything) | spellF1 0.674 |
| labels from BAR-ATH used for training, tuning or selection | **none** |
| hand labels of any kind | **none** |

## 2. Inputs: what we used and what we dropped

| input | used? | how |
|---|---|---|
| ATD tracking (tactical feed) | **yes** | the only per-frame input: player and ball `x, y`. Speeds are recomputed from positions. |
| ATD metadata | **yes** | team of each slot, goalkeeper slots, 1st-half attack direction. |
| Smart Tagging | **partly** | only the automatic `Kick Off (Start)` intervals, to gate the kickoff detector. Restart-team hints were tested and dropped (§5.11). Phase tags were not used. |
| ATD speed channel | no | in m/s despite the XML saying kph. The ball speed is present on only 8.7 % of frames with a ball position, and implausibly low, so we compute velocity from positions. |
| homography, videos | no | not needed by the final pipeline. We watched the videos to understand the data, which the rules allow. Ball re-detection from the video was considered (§7) but not built. |
| public 5-minute ground truth | sanity only | to check frame alignment and file format. Never used to train, tune or select. |

The two facts that shaped everything else (from `analysis/00`–`02`):

- **ATD has only `x, y` (and an unreliable speed).** No ball height, no alive/dead flag and no owner,
  so D and H/A have to be inferred from geometry and motion.
- **The ball is missing on ≈41 % of graded frames, in long blocks** (median gap 10.7 s; 99.7 % of
  missing frames are in gaps longer than 5 s). Interpolation can't cover that. About 80 % of the
  missing frames sit in gaps that show a stoppage signature (players slow down), so most of the
  blackout problem is a dead-ball problem.

## 3. External data (other games only)

### IDSSE / DFL: training and validation

- **What:** 7 professional German matches (Bundesliga 1 and 2) with 25 fps tracking and event data.
  Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer",
  *Scientific Data* (2025). Licence CC BY 4.0. Public mirror
  <https://huggingface.co/datasets/pysport/idsse-data>, loaded with kloppy 3.18.0. How to get it:
  [`docs/idsse-and-dead-ball-model.md`](docs/idsse-and-dead-ball-model.md).
- **Who labelled it:** the per-frame alive/dead flag (`BallStatus`) and team in possession
  (`BallPossession`) are part of the DFL provider's tracking feed. **We labelled nothing.**
- **How much:** all 7 matches, about 44.8 % dead frames and 713 dead spells in total.
- **How we use it:** to train both models, and to choose every parameter by leave-one-match-out
  cross-validation. Before training, the clean DFL tracking is **degraded to look like our ATD**
  ([`dead_ball/degrade.py`](dead_ball/degrade.py)):
  - the ball is hidden in blocks whose rate, length and placement were fitted to BAR-ATH without
    labels (§5.7);
  - the ball is smoothed with a 0.25 s Gaussian, which matches the ATD ball's motion (§5.6);
  - ball height is dropped, speeds come from positions, and players drop out at random;
  - DFL freezes the ball while it is dead. We replace that with realistic movement (overshoot,
    carries, throws, pauses), because the first model learned to key on the frozen ball (§5.3).

### Metrica Sports sample games 1–3: validation only

- **What:** Metrica Sports' public sample data, <https://github.com/metrica-sports/sample-data>:
  anonymised matches with tracking and event data. These are not the competition match.
- **Who labelled it:** the events are Metrica's. We derive per-frame labels from them with a fixed
  recipe ([`dead_ball/metrica.py`](dead_ball/metrica.py)): dead from the stoppage to the set
  piece; owner = team of the last on-ball event, carried forward. **We labelled nothing.**
- **How we use it:** validation only, never training. We used it to check that DFL's
  dead-ball and possession conventions don't differ from Metrica's in a way that would change our
  choices (§5.2, §5.10), and to test the kickoff detector (§5.11).

### Borrowed method

- **Vidal-Codina et al. (2022)**, "Automatic event detection in football using tracking data",
  *Sports Engineering* 25. We ported its player-possession rule from databallpy 0.8.1 (MIT,
  <https://github.com/alek050/databallpy>) into
  [`possession/vidal_codina.py`](possession/vidal_codina.py). We wrote our own port because
  installing databallpy would have downgraded numpy and pandas in our pinned environment. In the
  final pipeline, the rule's outputs are only **features** of the learned team model.

## 4. The final pipeline and its possession rules

### 4.1 Dead ball (D)

- **Model:** `HistGradientBoostingClassifier` (scikit-learn, fixed seed) on windowed per-frame
  features computed from `x, y` only ([`dead_ball/features.py`](dead_ball/features.py)): team mean
  speed over the past/next seconds, player stillness, team spread and shape, ball distance to the
  touch and goal lines, time since and until the ball was last seen, players near corners, in the
  box or in the centre circle, and goalkeeper speed and position. Ball speed is deliberately left
  out (§5.4).
- **Training:** all 7 IDSSE matches, 2 degraded copies each, with the DFL flag as the target, used
  as given (no time shift).
- **Decoding:** hysteresis on P(dead): enter D above 0.7, leave below 0.3, and drop dead spells
  shorter than 3 s. These were chosen by inner 3-fold CV by match.
- **What it relies on** (permutation importance, held out): team mean speed over 3 s (P(dead)
  crosses 0.5 at about 1.45 m/s), ball distance to the line (crosses just over the line),
  players' speed over the next 3 s, and time since the ball was last seen.

**Where we draw the line:** D starts when the ball is out (as soon as its ground projection leaves
the pitch, which DFL's flag marks a median 0.16 s later) or at the whistle for fouls. D ends on
the restart touch. That's the convention we measured in DFL's flag (§5.2). The competition defines
D as whatever its corrected tracking marks dead, and DFL's flag is the closest labelled equivalent
we could find on other games.

### 4.2 Team in possession (H / A)

- **Per-frame evidence:** `HistGradientBoostingClassifier` estimating P(home has the ball) for
  every live frame ([`possession/learned.py`](possession/learned.py)). It uses positions only and
  looks both backwards and forwards in time:
  - nearest home and away player distances to the ball, now, before and after;
  - the Vidal-Codina rule's zone runs and accepted gains (possession zone 2.0 m, angle 10°);
  - team and ball velocity relative to each team's direction of attack;
  - team shape;
  - time since the ball was last seen.
- **Training:** every 5th live frame of 2 degraded, smoothed copies of each IDSSE match, with
  DFL's `BallPossession` as the target. Every row is duplicated with the teams swapped, and
  predictions are symmetrised, so the model can't learn a home bias.
- **Decoding:** inside each predicted live stretch, a 2-state Viterbi picks the team sequence and
  pays a fixed cost of **6** (in log-probability units) per team switch. Ownership never carries
  across a predicted stoppage or the half-time break.

**Where we draw the line:**

- **Loose balls, passes in flight and blackouts** keep the last team. The decoder only switches
  when the evidence for the other team outweighs the switch cost.
- **Challenges and deflections:** a momentary touch by the other team isn't enough to switch. It
  has to be sustained, and the cost of 6 sets how much. That's close to Metrica's "controlled
  possession" (§5.10).
- **Restarts:** the dead spell ends on the restart touch, and the team that restarts is whatever the
  evidence says from that frame on. No set-piece rule is hard-coded.

### 4.3 Opening kickoff

- **Detector** ([`dead_ball/kickoff.py`](dead_ball/kickoff.py)): under Law 8, every player stands in
  his own half at a kickoff. It looks for the two teams in opposite halves (2 m tolerance, one
  outlier player allowed, gaps up to 1 s closed) for at least 3 s, followed by 20 s in which that
  formation is gone. The kick is placed where the formation breaks, minus a 0.6 s lag measured on
  other games.
- **Gate:** a detection counts only within ±5 s of Smart Tagging's automatic `Kick Off (Start)`
  interval for that half. Everything in a graded window before the accepted kick is D. The code
  holds **no frame or timestamp constants** of this match.

### 4.4 Output

H/A/D inside the two graded windows (video frames 11,185–90,651 and 110,808–190,197) and X
everywhere else. The submission row is the video frame, which equals ATD `frame_id − 1`.

## 5. The journey: what we tried, what failed, what we kept

Everything below was scored on other games (held-out IDSSE unless stated). BAR-ATH appears only in
label-free checks: aggregates of our own predictions compared with the same aggregates on IDSSE.
The detailed write-ups are in [`observations/`](observations/README.md), and each decision with its
reason is in [`docs/decisions.md`](docs/decisions.md).

### 5.1 Day one: what the data is (`analysis/00`–`02`)

We started with aggregate-only probes of the ATD and the official docs. The result was the picture
in §2: no flags, no ball height, an unreliable speed channel, and ball blackouts in long blocks that
mostly coincide with stoppages. When the ball vanishes, the team-wide mean speed halves (2.0 →
1.0 m/s median). That made **D detection the first job**, because it also solves most of the
missing-ball problem. Only about 15 % of the missing frames are in gaps with no restart cue, and
those are the real "possession changed while we were blind" cases.

### 5.2 Finding labelled data from other games (`analysis/03`–`05`)

- **IDSSE has exactly the labels we need:** a per-frame alive/dead flag and owning team, at the
  same 25 fps. One trap: kloppy's loader drops dead frames by default (`only_alive=True`).
- **DFL's convention:** the flag goes dead a median 0.16 s after the ball crosses the line (95.5 %
  within 0.5 s), close to the whistle for fouls, and alive on the restart (98 % of set pieces within
  5 s). We train on the flag **as given, with no shift**.
- **DFL vs Metrica:** we put both providers' events through the same recipe. Stoppage durations
  by restart type agree to within a few seconds, so there's no gross convention difference.
- To understand the target convention itself, we **watched** the public 5-minute window with its
  ground truth in a local viewer (allowed: "watching the video … to understand them"). No offset,
  threshold or model choice was taken from that window. The viewer isn't part of this
  repository.

### 5.3 Dead-ball classifier: model vs hand rules (`analysis/06`, `07`)

- **Kept:** the gradient-boosted model. With DFL's own team labels (to isolate D), held-out
  spellF1 goes from **0.768 with no D to 0.881** with the model. Hand rules distilled from the
  model's read-out reach 0.836, and the model wins 7/7 matches by 0.038–0.049. That's more than
  the 0.02 margin we had agreed for choosing a learned model over rules.
- **Failed, then fixed: a data leak.** DFL freezes the ball while dead (100 % of dead frames have
  an exactly still ball, against 0.8 % of alive frames and 0.1 % of our ATD's). The first model
  keyed on it and scored 0.905. We discarded that model, and the degradation now moves the dead ball
  realistically.

### 5.4 Does it transfer to our ATD? (`analysis/08`)

We compared BAR-ATH features with degraded-IDSSE features, label-free. **Player signals
transfer; the ball doesn't.** The model chopped BAR-ATH stoppages up (111 per 90 minutes, median
12.0 s, against 96 per 90 minutes and 19.6 s in IDSSE truth).

| variant | IDSSE spellF1 (DFL teams) | BAR-ATH stoppages / 90 min | BAR-ATH median stoppage |
|---|---|---|---|
| original (ball speed + straight dead-ball carry) | **0.881** | 111 | 12.0 s |
| no ball-speed feature | 0.861 | 100 | 18.4 s |
| realistic ("mixed") dead-ball motion | 0.870 | 100 | 15.0 s |
| **both (kept)** | 0.864 | 101 | 17.6 s |

We gave up about 0.02 on IDSSE for a model whose behaviour on our match looks like real football.

### 5.5 Team logic, first round (`analysis/09`)

With our D in place, the team logic was the bottleneck. Held-out spellF1, raw DFL ball:

| team logic | spellF1 | outcome |
|---|---|---|
| teammate's first pipeline ("v1", nearest-team rules) | 0.279 | dropped: finds only about 35 % as many possessions as the truth |
| teammate's two-stage geometry decoder ([`baselines/`](baselines/two_stage_possession.py)) | 0.331 | dropped: about the right number of possessions, but its 0.24 s switch confirmation flips team whenever the other team is nearer for a moment |
| **Vidal-Codina rule, tuned** (zone 1.0 m) | **0.523** | kept, wins 7/7 matches |
| *DFL's own owner (ceiling for our D)* | *0.864* | |

A teammate also built rule variants and a learned owner model in parallel. The learned-owner idea
came back later, in a stronger form, as the final team logic (§5.9).

### 5.6 Our ball is smoothed (`analysis/11`)

The ATD ball's frame-to-frame speed changes match the DFL ball after a σ = 0.25 s Gaussian (q90/q99
0.61/1.26 m/s against the ATD's 0.68/1.19; the raw DFL ball gives 1.34/12.0). On a smoothed ball the
rule's speed trigger never fires, and the earlier tuning collapsed (0.523 → 0.31). Re-tuned on
smoothed IDSSE: zone 2.5 m. **The honest expectation fell to 0.374.** VC still beat geometry
(0.298) and v1 (0.279) in 7/7 matches. **Failed:** a window-based speed trigger built for the
smoothed ball. It added false gains.

### 5.7 Blackouts calibrated to BAR-ATH (`analysis/12`)

**Failed assumption:** our first degradation hid 31 % of live play in long random gaps. BAR-ATH,
measured label-free with our own D calls, hides only **10.6 % of live frames** (76.7 % of dead
ones). Gaps start near the stoppage, end just before the restart, and live gaps are 1.9× more likely
after a duel. The new `calibrated` degradation reproduces this (12.3 % / 70.7 %; duel ratio
1.69 against 1.67). Retrained on it: D alone reaches 0.900, and VC rose to 0.438 with a 2.0 m
zone. The **one-command submission** was built at this point.

### 5.8 Error anatomy (`analysis/13`)

Where did VC lose? **Not at boundaries:** shifting team switches or D edges gains at most +0.006,
so it wasn't adopted. **Short possessions:** 60 % of true possessions last under 5 s and VC catches
20 % of them. It misses about 129 short opponent spells per match and invents about 85 false flips.
No threshold fixes both at once. That pointed to per-frame evidence plus a decoder with a switching
cost.

### 5.9 Learned team logic (`analysis/14`, `15`), kept

The P(home) model with the switch-cost Viterbi (§4.2) reaches **0.601 against VC's 0.441, better in
7/7 matches** (minimum +0.104). Switch cost 6 was chosen in 7/7 folds and sits on a plateau
(4–8 score 0.598–0.601). Free switching gives only 0.459, so the decoder does much of the work.
On BAR-ATH, label-free, its aggregates sit on top of held-out IDSSE: 6.19 against 6.25
possessions per live minute, and a median possession of 5.3 s against 5.5 s.

### 5.10 Is DFL's possession convention Metrica's? (`analysis/16`)

The worry was that DFL counts many very short possessions and Metrica's "controlled possession"
counts fewer, which would call for a higher switch cost. Through the same event recipe on both
providers, Metrica is **not stricter**: its best cost is 6 (ATD-like ball), against 12 on DFL.
**Rule fixed before running:** change the cost only if 6 scores more than 0.02 below Metrica's best
and that best differs from DFL's. On Metrica, cost 6 is 0.000 below its best, so the cost stays at 6.
Trained on DFL only, the model scores 0.77 against Metrica's event possessions (VC 0.52).

### 5.11 Opening kickoff and Smart Tagging (`analysis/17`–`19`)

- **Problem:** the 2nd half is graded from about 117 s before its kickoff, and all of that is D. Our
  D model called only about 58 % of it D.
- **Kept:** the Law-8 detector (§4.3). It finds 32/32 testable kickoffs on IDSSE and Metrica, all
  within 1 s, with no false formations in 369 minutes of live play. Gated by Smart Tagging, it
  turns 49.1 s of invented live play into D. The 2nd-half pre-kickoff stretch is now 100 % D.
- **Failed:** without the gate, the 1st half (graded from the kick itself) latched onto a kickoff
  after a goal. The gate fixed that.
- **Tested and dropped:** Smart Tagging's set-piece team as the team at the start of the next live
  stretch. On held-out IDSSE, even perfect hints add nothing (0.601 → 0.601), because the model
  already starts 95 % of those stretches with the right team. Realistic tag errors lose in 7/7
  matches. Our pre-set adoption rule (+0.003 in ≥ 5/7) wasn't met.
- **Not tried:** phase tags as a team prior. Their weight could only be chosen on BAR-ATH labels.

### 5.12 Scoreboard of the main steps (held-out IDSSE, spellF1)

| step | team logic | spellF1 |
|---|---|---|
| raw DFL ball, first D model | VC (zone 1.0 m) | 0.523 *(optimistic: unsmoothed ball)* |
| ATD-like smoothed ball | VC (2.5 m) | 0.374 |
| + blackouts calibrated to BAR-ATH, D retrained | VC (2.0 m) | 0.438–0.441 |
| **+ learned P(home) and switch-cost Viterbi (submitted)** | learned | **0.601** |
| *same, with perfect D* | learned | *0.667* |

## 6. Guard rails

- **No hand labels** on BAR-ATH, in any amount or for any purpose. No labels of BAR-ATH from any
  source: no event feeds, no corrected tracking, no private ground truth.
- **The public 5 minutes are a sanity check only:** frame alignment, file format, label meaning.
  Every number from that window in our notes is marked "sanity only", and no parameter, threshold
  or model choice was made on it.
- **No frame or timestamp constants** of this match. The kickoff gate uses Smart Tagging's own
  output, which is an algorithmic input.
- **Adoption rules were written down before the runs** that decided them (§5.10, §5.11).
- **Deterministic:** fixed seeds and a pinned environment (`uv.lock`). A from-scratch rebuild,
  including retraining, gives the same file byte for byte.

## 7. Limitations and what we'd do with more time

- **Short possessions are still the main loss.** About 30 % of DFL's true possessions last under
  2 s. Some of these are probably not possessions under Metrica's convention, but the real ones
  that we miss cost spellF1.
- **Convention risk.** Everything was trained on DFL's flags. We checked dead-ball timing and
  possession granularity against Metrica's events, but a sub-second systematic offset between
  DFL's flag and Metrica's corrected-tracking flag can't be measured from other games.
- **Live-play ball blackouts** (about 10 % of live frames on BAR-ATH) leave the team model with only
  player positions. Possessions that both start and end inside a blackout are recalled poorly
  (0.18 against 0.44 with the ball visible).
- **Home share.** On BAR-ATH the model gives the home side 61 % of H/A time, against 52.8 % on
  IDSSE. That may be real for this match, but we can't check it without labels.
- **The kickoff gate depends on Smart Tagging** finding the opening kickoff. If the tag is missing,
  the half simply runs without the forced D.
- **With more time:** re-detect the ball from the tactical video (a public ball detector, mapped
  to the pitch with the provided homography) to fill live blackouts; add more labelled games from
  other providers; and build a touch detector from player acceleration around the ball.

## 8. Reproduction

Requirements: [uv](https://docs.astral.sh/uv/), the competition data bundle, and the IDSSE matches.

```bash
uv sync                                   # Python 3.12, deps pinned in uv.lock
# competition data → data/tactical/ and data/broadcast/ (see data/README.md)
# IDSSE (7 matches) → data/external/idsse/   (see docs/idsse-and-dead-ball-model.md)
uv run python scripts/make_submission.py --retrain --out submissions/LaMacIA.csv
uv run python evaluation/validate_submission.py submissions/LaMacIA.csv
```

The script trains both models from scratch (`--retrain`; without it, cached models in
`artifacts/submission/` are reused), predicts both graded windows, writes the CSV, prints its
sha256 and runs the official validator. The submitted file has sha256
`0add77a7e45b59b1619c82d3c4985ff63b483abf35b56d4536fb20bd6fa8e2ed`; a from-scratch rebuild in a
fresh clone (about 10 minutes, including retraining) reproduces it exactly. Metrica sample games are needed only for the validation
scripts in `analysis/`, not for the submission. Unit tests: `uv run python -m pytest -q`.
