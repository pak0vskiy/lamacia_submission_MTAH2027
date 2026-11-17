# Dead-ball model: getting the data and training it

This guide takes you from a fresh clone to the trained dead-ball models. For background (what the
model is, its features and results), read [`dead_ball/README.md`](../dead_ball/README.md).

> **No model files are committed.** Everything trained ends up in `artifacts/dead_ball/`, which
> is gitignored. Everyone rebuilds it locally with the steps below. Seeds are fixed, so the result
> is deterministic for the same data and pinned dependencies.

## 0. Prerequisites

- [uv](https://docs.astral.sh/uv/) installed. The env is Python 3.12, with deps pinned in
  `pyproject.toml` + `uv.lock` (kloppy 3.18.0, scikit-learn 1.9.1, …).
- A clone of this repo. Run everything from the repo root.

```bash
uv sync
```

## 1. Get the training data (IDSSE / DFL)

**What it is:** 7 professional German matches (Bundesliga 1 and 2) from the DFL.
- Tracking at 25 fps, with a per-frame **alive/dead flag** (`BallStatus`) and **team in
  possession** (`BallPossession`).
- Official event data.

**Source:** Bassek et al., "An integrated dataset of spatiotemporal and event data in elite
soccer", *Scientific Data* (2025). Licence **CC BY 4.0**.

**Public mirror:** <https://huggingface.co/datasets/pysport/idsse-data> (21 XML files, flat layout).

The code expects the XML files directly in **`data/external/idsse/`**. `data/` is gitignored,
so never commit them.

### Option A: Hugging Face CLI (no install needed, via uvx)

```bash
uvx --from huggingface_hub hf download pysport/idsse-data \
    --repo-type dataset --include "DFL_*.xml" --local-dir data/external/idsse
```

### Option B: plain curl (bash)

```bash
mkdir -p data/external/idsse && cd data/external/idsse
BASE=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
for m in 000001_DFL-MAT-J03WMX 000001_DFL-MAT-J03WN1 \
         000002_DFL-MAT-J03WOH 000002_DFL-MAT-J03WOY 000002_DFL-MAT-J03WPY \
         000002_DFL-MAT-J03WQQ 000002_DFL-MAT-J03WR9; do
  for kind in 02_01_matchinformation 03_02_events_raw 04_03_positions_raw_observed; do
    curl -sSL -O "$BASE/DFL_${kind}_DFL-COM-${m}.xml"
  done
done
cd -
```

### Check

You should have 21 files: 3 per match × 7 matches.

```
data/external/idsse/
  DFL_02_01_matchinformation_DFL-COM-00000X_DFL-MAT-<id>.xml      # metadata: pitch, players, positions
  DFL_03_02_events_raw_DFL-COM-00000X_DFL-MAT-<id>.xml            # events (used only by analysis/04-05)
  DFL_04_03_positions_raw_observed_DFL-COM-00000X_DFL-MAT-<id>.xml # tracking + flags (big files)
```

Match ids: `J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9`.

```bash
uv run python -c "from dead_ball.idsse import MATCH_IDS; print(len(MATCH_IDS), MATCH_IDS)"
# expected: 7 ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9']
```

`MATCH_IDS` is built from the files on disk. If it prints fewer than 7, some position files are
missing.

## 2. Run the unit tests (optional, no data needed)

```bash
uv run python -m pytest tests/dead_ball -q
```

These use synthetic periods (`tests/dead_ball/conftest.py`), so they pass even without IDSSE.

## 3. Train

```bash
uv run python analysis/06_dead_ball_cv.py
```

This trains and evaluates the **chosen configuration**. The defaults live in `dead_ball/cv.py`:

| setting | value |
|---|---|
| features | everything in `dead_ball/features.py` except `ball_spd_1` (`DEFAULT_DROP`) |
| degradation | `"mixed"` dead-ball motion (`DEFAULT_MOTION`), 41 % ball blackouts, 2 % player drop-out |
| training copies | 2 degraded copies per match (`--copies 2`), blackout–stoppage coincidence c ~ U[0, 0.9] |
| rows | every 5th frame (`SUBSAMPLE = 5`) |
| classifier | `HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False)` |
| decoder | hysteresis enter/exit + min dead spell, picked by inner 3-fold CV from enter {0.5,0.6,0.7} × exit {0.3,0.4,0.5} × min_dead_s {1,2,3} |
| validation | leave-one-match-out over the 7 matches |

What happens, step by step:
1. **Load + cache.** Each match is parsed with kloppy
   (`sportec.load_tracking(only_alive=False, coordinates="sportec")`) and cached as
   `artifacts/idsse_match_<id>.npz`. This is the slow part on the first run. After that, it's read
   from the cache.
2. **Degrade + features.** Training copies plus test copies (c = 0, 0.5, 0.9) are built for each
   match and cached in `artifacts/dead_ball/prepared_v3mixed_<id>_2.joblib`.
3. **7 folds.** For each held-out match:
   - tune the decoder by inner CV on the other 6;
   - fit the classifier on all 6;
   - score the held-out match in settings A (DFL teams) and B (Sam's v1 teams).
   
   The v1-as-is and no-D references are scored too.
4. **Print** mean and std tables of spellF1 / macroF1 / stoppage P/R, plus the decoder choices.
   Aggregates only.

### Outputs (all in `artifacts/dead_ball/`, gitignored)

| file | what |
|---|---|
| `fold_model-no-ball_spd_1@mixed_<match>.joblib` | **the trained models**: one per held-out match, each a dict with `fitted` (the sklearn classifier) and `params` (its decoder settings) |
| `oof_model-no-ball_spd_1@mixed_<match>.joblib` | out-of-fold dead probabilities on the held-out match |
| `cv_model-no-ball_spd_1@mixed.csv` | per-match scores |

### Expected numbers (sanity check that your run matches ours)

Mean over the 7 held-out matches at c = 0.5:
- spellF1 about **0.864** in setting A (no-D floor 0.768);
- about **0.279** in setting B (v1 as is 0.218).

Small differences can come from a different scikit-learn build. Large ones mean the data or the
config differs.

### Useful flags

```bash
uv run python analysis/06_dead_ball_cv.py --matches 3          # quick smoke test on 3 matches
uv run python analysis/06_dead_ball_cv.py --bootstrap 200      # spellF1 confidence intervals
uv run python analysis/06_dead_ball_cv.py --method rules       # hand-rule baseline instead of the model
uv run python analysis/06_dead_ball_cv.py --drop               # keep every feature (incl. ball speed)
uv run python analysis/06_dead_ball_cv.py --motion carry       # older straight-carry degradation
uv run python analysis/06_dead_ball_cv.py --tag _try1          # suffix on the output CSV
```

Changing `--drop` or `--motion` changes the model name: `model`, `model-no-<cols>`, plus
`@mixed` when motion isn't carry. Fold files from different configs don't overwrite each other.

## 4. Inspect what it learned (optional)

```bash
uv run python analysis/07_dead_ball_patterns.py
```

This prints held-out permutation importance, partial-dependence thresholds and a depth-4
surrogate tree. It needs step 3 first.

## 5. Use the models on our match (BAR-ATH)

```bash
uv run python analysis/08_atd_vs_idsse_features.py
```

This needs the competition ATD in `data/` (see `DATA.md` / `starter/`).

It shows how to apply the models:
- build a `Period` from tracking;
- call `frame_features(period, 25.0, half_len, half_wid)`;
- **average `predict_proba` of the 7 fold models** (dropping `ball_spd_1`);
- decode with `decode_dead` using the most common fold decoder params.

It prints only aggregates: dead share, stoppage count and lengths. No BAR-ATH labels are used.

> **There is no single "final" model yet.** For now, the deployable model is the ensemble of the 7
> fold models. A model trained on all 7 matches, built inside `scripts/make_submission.py`, is a
> planned next step.

## Troubleshooting

- **`MATCH_IDS` is empty / `IndexError` in `dfl_files`:** the XMLs aren't in
  `data/external/idsse/`, or they're in a subfolder. Move them up to that exact folder.
- **A stale cache after changing the degradation or features:** delete the
  `artifacts/dead_ball/prepared_*.joblib` files (and `fold_*`/`oof_*` for that config) and rerun.
  The `artifacts/idsse_match_*.npz` files only depend on the raw data and can stay.
- **Memory:** position files are large, but each match is parsed once and cached. If parsing runs
  out of memory, close other programs and rerun. Matches already cached are skipped.
- **Windows:** use Git Bash for the curl loop, or option A, which works in any shell.

## Rules reminder

- Never commit `data/` or `artifacts/`. Don't print per-frame data: scripts print aggregates only.
- IDSSE is external labelled data from **other games**. It's documented in
  `REPORT.md`, which is allowed under the competition rules.
- No BAR-ATH labels, including the public 5 min, are used to train or select anything.
