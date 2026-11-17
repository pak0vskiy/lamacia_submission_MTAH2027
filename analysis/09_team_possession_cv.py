"""Team logic head-to-head on held-out IDSSE, scored with the official spellF1. Aggregates only.

    uv run python analysis/09_team_possession_cv.py [--matches N]

Every team logic sits on the same D: our dead-ball model's fold model for that match (trained
without it), plus an oracle-D run to isolate the team part. Test copies are rebuilt
bit-for-bit from analysis/06's seeds; positions are needed, and those copies were cached
without them.

Team logics:
- dfl:   DFL's own owner (= setting A; upper reference)
- v1:    Sam's v1 (= setting B)
- geom:  Sam's two-stage geometry decoder (baselines/two_stage_possession.py, default OwnerConfig)
- vc:    Vidal-Codina 2022 rule (possession/vidal_codina.py), paper/databallpy defaults
The ball the team logics see is smoothed like our ATD's (--ball-smooth, default 0.25 s).
- vc*:   Vidal-Codina with parameters chosen leave-one-match-out on the other 6 matches (c=0.5)
Other games only; nothing from BAR-ATH is read.
"""

import argparse
import itertools
import os
import sys
import warnings
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dead_ball import cv  # noqa: E402
from dead_ball.degrade import degrade, smooth_ball  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.pipeline import A, D, H, X, concat, decode_dead, oracle_codes, score, truth_codes  # noqa: E402
from dead_ball.v1_adapter import v1_codes_from_frame  # noqa: E402
from possession.vidal_codina import VCParams, period_codes  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "baselines"))
import two_stage_possession as sam  # noqa: E402

MODEL = "model-no-ball_spd_1@mixed"
OUT = os.path.join(cv.OUT, "..", "possession")
GRIDS = {
    # first pass on the smoothed ball
    "wide": dict(pz_radius=(0.75, 1.0, 1.5, 2.0), bv_threshold=(1.0, 2.0, 3.5, 5.0),
                 ba_threshold=(10.0, 20.0, 45.0), window_frames=(1, 3, 6, 12)),
    # second pass: "wide" peaked at its edges (largest zone, smallest angle)
    "refine": dict(pz_radius=(1.5, 2.0, 2.5, 3.0, 4.0), bv_threshold=(5.0, 7.5, 10.0),
                   ba_threshold=(5.0, 10.0), window_frames=(1, 3, 6)),
}
# Gaussian sigma (s) that makes the degraded IDSSE ball's speed changes match our ATD's
# (analysis/11_ball_smoothing.py). Applied to the ball the team logics see; D is unchanged.
BALL_SMOOTH_S = 0.25
COLS = ["spell_f1_mean", "macro_f1", "count_ratio"]


def rebuild(m, i, j, c, blackout="coincidence"):
    """Degraded periods of test copy (match index i, coincidence index j) as in cv.prepare."""
    rng = np.random.default_rng([cv.SEED, i, 100 + j])
    return [degrade(p, m.fps, rng, c, motion=cv.DEFAULT_MOTION, blackout=blackout) for p in m.periods]


def geom_codes(q, dead, fps):
    cfg = sam.OwnerConfig()
    lab, _ = sam.decode_owners(sam.ownership_evidence(q, cfg), dead, fps, cfg)
    return np.select([lab == "H", lab == "A", lab == "D"], [H, A, D], X).astype(np.int8)


def row(method, dmode, mid, c, gt, pred):
    r = score(concat(gt), concat(pred))
    return dict(method=method, d=dmode, match=mid, coincidence=c, **r,
                count_ratio=r["n_pred_spells"] / r["n_gt_spells"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=None)
    ap.add_argument("--ball-smooth", type=float, default=BALL_SMOOTH_S, help="sigma in s; 0 = raw DFL ball")
    ap.add_argument("--tag", default="", help="suffix for the output CSVs")
    ap.add_argument("--grid", default="refine", choices=list(GRIDS))
    ap.add_argument("--blackout", default=cv.DEFAULT_BLACKOUT, choices=["coincidence", "calibrated"],
                    help="calibrated: BAR-ATH-fitted blackouts and the matching '+cal' fold models")
    ap.add_argument("--d-model", default=None, help="fold model name to use for D (default: matches --blackout)")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    ids = cv.default_ids(a.matches)
    data = cv.prepare(ids, 2, cv.DEFAULT_MOTION, a.blackout)
    model_name = a.d_model or MODEL + ("+cal" if a.blackout == "calibrated" else "")
    G = GRIDS[a.grid]
    grid = [VCParams(**dict(zip(G, v))) for v in itertools.product(*G.values())]
    rows, grid_rows = [], []
    for i, mid in enumerate(ids):
        # joblib = pickle: only our own fold models, written locally by dead_ball/cv.py
        fold = joblib.load(os.path.join(cv.OUT, f"fold_{model_name}_{mid}.joblib"))
        model = cv.Model(drop=cv.DEFAULT_DROP)
        m = load_match(mid)
        for j, c in enumerate(cv.TEST_COINCIDENCE):
            copy = data[mid]["test"][c]
            qs = rebuild(m, i, j, c, a.blackout)
            for q, f in zip(qs, copy.feats):       # rebuilt copy must equal the cached one
                assert np.array_equal(np.isfinite(q.ball[:, 0]), f.ball_vis.to_numpy() > 0)
            qs = [replace(q, ball=smooth_ball(q.ball, m.fps, a.ball_smooth)) for q in qs]
            gt = [truth_codes(q) for q in qs]
            pred_dead = [decode_dead(model.predict(fold["fitted"], f), m.fps, **fold["params"])
                         for f in copy.feats]
            for dmode, deads in (("model", pred_dead), ("oracle", [q.dead for q in qs])):
                rows.append(row("dfl", dmode, mid, c, gt, [oracle_codes(q, d) for q, d in zip(qs, deads)]))
                rows.append(row("v1", dmode, mid, c, gt,
                                [v1_codes_from_frame(f1, data[mid]["match"], d.astype(float))
                                 for f1, d in zip(copy.v1, deads)]))
                rows.append(row("geom", dmode, mid, c, gt, [geom_codes(q, d, m.fps) for q, d in zip(qs, deads)]))
                rows.append(row("vc", dmode, mid, c, gt, [period_codes(q, d, m.fps) for q, d in zip(qs, deads)]))
                if c == 0.5:
                    for k, prm in enumerate(grid):
                        r = row("vc_grid", dmode, mid, c, gt, [period_codes(q, d, m.fps, prm) for q, d in zip(qs, deads)])
                        grid_rows.append(dict(r, k=k))
        print(f"  done {mid}", flush=True)
        del m

    df = pd.DataFrame(rows)
    g = pd.DataFrame(grid_rows)
    # vc*: for each held-out match, the grid point with the best mean spellF1 on the other matches.
    for dmode in ("model", "oracle"):
        gd = g[g.d == dmode]
        for mid in ids:
            best = gd[gd.match != mid].groupby("k").spell_f1_mean.mean().idxmax()
            r = gd[(gd.match == mid) & (gd.k == best)].iloc[0].to_dict()
            r.update(method="vc*", chosen=str(grid[best]))
            rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"team_cv{a.tag}.csv"), index=False)          # per-match aggregates, local
    g.to_csv(os.path.join(OUT, f"team_cv_vc_grid{a.tag}.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"\n{len(ids)} held-out matches, ball smoothing sigma {a.ball_smooth} s. D = our model's fold model ('model') or the DFL flag ('oracle').")
    for dmode in ("model", "oracle"):
        t = (df[df.d == dmode].groupby(["method", "coincidence"])[COLS].mean()
             .unstack("coincidence").round(3))
        print(f"\nmean over held-out matches, D = {dmode}")
        print(t.to_string())
    print("\nstd of spellF1 over matches (c=0.5)")
    print(df[df.coincidence == 0.5].groupby(["d", "method"]).spell_f1_mean.std().round(3).unstack("d").to_string())
    ch = df[df.method == "vc*"].groupby(["d", "chosen"]).size()
    print("\nvc* parameters chosen per fold:\n" + ch.to_string())
    top = g[g.d == "model"].groupby("k").spell_f1_mean.mean().sort_values(ascending=False).head(5)
    print("\nVC grid, D = model, c = 0.5, top 5 by mean spellF1 over all matches (descriptive only):")
    for k, v in top.items():
        print(f"  {v:.3f}  {grid[k]}")


if __name__ == "__main__":
    main()
