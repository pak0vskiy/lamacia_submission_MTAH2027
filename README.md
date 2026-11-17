# LaMacIA: ball possession for More than a Hack 2027 (Phase 1)

Frame-level ball possession (H / A / D) for the BAR-ATH match, from tracking data only, with every
model trained and validated on other games.

**Read [`REPORT.md`](REPORT.md)** for the approach, the data we used, the possession rules, everything
we tried along the way (including what failed), and the limitations.

## Reproduce the submission

```bash
uv sync
uv run python scripts/make_submission.py --retrain --out submissions/LaMacIA.csv
uv run python evaluation/validate_submission.py submissions/LaMacIA.csv
```

The rebuild is deterministic (fixed seeds, pinned `uv.lock`). The script prints the file's sha256.

### Data you need (not in this repository)

| what | where | how to get it |
|---|---|---|
| competition bundle (ATD, Smart Tagging, homography, videos) | `data/tactical/`, `data/broadcast/` | enrolment; see [`data/README.md`](data/README.md) |
| IDSSE / DFL, 7 matches (CC BY 4.0) | `data/external/idsse/` | [`docs/idsse-and-dead-ball-model.md`](docs/idsse-and-dead-ball-model.md) |
| Metrica sample games 1–3 (validation scripts only) | `data/external/metrica/` | files from <https://github.com/metrica-sports/sample-data>, copied flat |

Models, caches and per-frame outputs go to `artifacts/` and `submissions/`, both gitignored. No match
data or frame-level file is committed.

## Layout

| path | what |
|---|---|
| `scripts/make_submission.py` | the one command: train, predict, write, validate |
| `dead_ball/` | dead-ball model: IDSSE loader, ATD-like degradation, features, CV, decoder, kickoff detector, Metrica loader |
| `possession/` | team logic: learned P(home) + switch-cost Viterbi; Vidal-Codina (2022) rule port |
| `analysis/00`–`19` | every experiment behind a decision, in order (aggregate output only) |
| `baselines/` | earlier team-logic baseline we compared against |
| `observations/` | one write-up per experiment, with numbers |
| `docs/decisions.md` | each decision and why, newest first |
| `evaluation/`, `starter/`, `games.py`, `competition/` | from the official competition repository (loaders, scorer, validator, rules) |
| `tests/` | unit tests on synthetic data: `uv run python -m pytest -q` |

## Licence

`evaluation/`, `starter/`, `games.py` and `competition/` come from the official competition
repository under its licence ([`LICENSE`](LICENSE)). That licence doesn't cover the match data.
