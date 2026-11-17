"""Learned team logic vs Vidal-Codina on held-out IDSSE (leave-one-match-out). Aggregates only.

    uv run python analysis/14_learned_team_cv.py [--matches N]

Same degradation (calibrated blackouts, 'mixed' dead-ball motion), the same 0.25 s ball smoothing
and the same D fold models as analysis/09 and 13. For each held-out match:
- the owner model (possession/learned.py) is trained on the other matches' training copies
  (live frames, DFL owner as target, every 5th frame);
- it predicts P(home) on the three test copies, which are decoded with a switch cost from a grid;
- the reported "learned*" picks the switch cost with the best mean spellF1 on the other matches.
Other games only; nothing from BAR-ATH is read.
"""

import argparse
import os
import sys
import warnings
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from dead_ball import cv  # noqa: E402
from dead_ball.degrade import degrade, smooth_ball  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.pipeline import A, D, H, X, decode_dead, truth_codes  # noqa: E402
from possession import learned  # noqa: E402
from possession.vidal_codina import TUNED, period_codes  # noqa: E402

D_MODEL = "model-no-ball_spd_1@mixed+cal"
BALL_SMOOTH_S = 0.25
SUBSAMPLE = 5
COSTS = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
OUT = os.path.join(cv.OUT, "..", "possession")


def test_periods(m, i, j, c):
    rng = np.random.default_rng([cv.SEED, i, 100 + j])
    return [degrade(p, m.fps, rng, c, motion=cv.DEFAULT_MOTION, blackout=cv.DEFAULT_BLACKOUT) for p in m.periods]


def feats(q, m):
    q = replace(q, ball=smooth_ball(q.ball, m.fps, BALL_SMOOTH_S))
    return q, learned.team_features(q.ball, q.home, q.away, m.fps, m.half_len, m.half_wid)


def spell_f1(gts, preds):
    sep = np.array([X], np.int8)
    g = np.concatenate([np.r_[sep, x] if i else x for i, x in enumerate(gts)])
    p = np.concatenate([np.r_[sep, x] if i else x for i, x in enumerate(preds)])
    runs = pm.valid_runs(g)
    r = pm.spell_scores(g, pm.fill_abstentions(p, runs), runs)
    return r["mean_f1"], r["n_pred"] / r["n_gt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    ids = list(cv.default_ids(a.matches))
    os.makedirs(OUT, exist_ok=True)

    # pass 1: training rows for every match (cached; our own local files)
    train = {}
    for i, mid in enumerate(ids):
        path = os.path.join(OUT, f"learned_train_{mid}.joblib")
        if not os.path.exists(path):
            m = load_match(mid)
            joblib.dump(learned.training_rows(m, i), path)
            del m
        train[mid] = joblib.load(path)
        print(f"  train rows {mid}: {len(train[mid][1])}", flush=True)

    # pass 2: leave one match out
    prepared = cv.prepare(ids, 2, cv.DEFAULT_MOTION, cv.DEFAULT_BLACKOUT)
    rows, acc_rows = [], []
    dmodel = cv.Model(drop=cv.DEFAULT_DROP)
    for i, mid in enumerate(ids):
        others = [x for x in ids if x != mid]
        clf = learned.fit(pd.concat([train[x][0] for x in others], ignore_index=True),
                          np.concatenate([train[x][1] for x in others]))
        joblib.dump(clf, os.path.join(OUT, f"learned_fold_{mid}.joblib"))
        fold = joblib.load(os.path.join(cv.OUT, f"fold_{D_MODEL}_{mid}.joblib"))   # our own fold model
        data = prepared[mid]
        m = load_match(mid)
        for j, c in enumerate(cv.TEST_COINCIDENCE):
            copy = data["test"][c]
            per = []
            for q, f in zip(test_periods(m, i, j, c), copy.feats):
                assert np.array_equal(np.isfinite(q.ball[:, 0]), f.ball_vis.to_numpy() > 0)
                qs, F = feats(q, m)
                p = learned.p_home(clf, F)
                dead = decode_dead(dmodel.predict(fold["fitted"], f), m.fps, **fold["params"])
                per.append(dict(q=qs, p=p, dead=dead, gt=truth_codes(q), vis=F.ball_visible.to_numpy() > 0))
                live = ~q.dead
                ok = (p >= 0.5) == q.home_owns
                for name, mask in (("visible", live & (F.ball_visible.to_numpy() > 0)),
                                   ("hidden", live & (F.ball_visible.to_numpy() == 0))):
                    acc_rows.append(dict(match=mid, copy=c, part=name, n=int(mask.sum()), correct=int(ok[mask].sum())))
            gts = [x["gt"] for x in per]
            for dmode in ("model", "oracle"):
                deads = [x["dead"] if dmode == "model" else x["q"].dead for x in per]
                f1, cr = spell_f1(gts, [period_codes(x["q"], d, m.fps, TUNED) for x, d in zip(per, deads)])
                rows.append(dict(method="vc", d=dmode, match=mid, copy=c, cost=np.nan, f1=f1, count_ratio=cr))
                for cost in COSTS:
                    f1, cr = spell_f1(gts, [learned.decode(x["p"], d, cost) for x, d in zip(per, deads)])
                    rows.append(dict(method="learned", d=dmode, match=mid, copy=c, cost=cost, f1=f1, count_ratio=cr))
        print(f"  done {mid}", flush=True)
        del m

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "learned_cv.csv"), index=False)    # per-match aggregates, local
    acc = pd.DataFrame(acc_rows).groupby("part")[["n", "correct"]].sum()

    pd.set_option("display.width", 200)
    print(f"\n{len(ids)} held-out matches x {len(cv.TEST_COINCIDENCE)} copies; ball smoothing {BALL_SMOOTH_S} s.")
    print("\nper-frame H/A accuracy on live frames (argmax, held-out):")
    print((acc.correct / acc.n).round(3).to_string())
    for dmode in ("model", "oracle"):
        d = df[df.d == dmode]
        vc = d[d.method == "vc"].groupby("match")[["f1", "count_ratio"]].mean()
        L = d[d.method == "learned"].groupby(["match", "cost"])[["f1", "count_ratio"]].mean()
        grid = L.f1.unstack("cost")
        picks, star = [], []
        for mid in grid.index:
            best = grid.drop(index=mid).mean().idxmax()
            picks.append(best)
            star.append(dict(match=mid, f1=grid.loc[mid, best], count_ratio=L.loc[(mid, best), "count_ratio"]))
        star = pd.DataFrame(star).set_index("match")
        print(f"\n=== D = {dmode} ===")
        print("learned, mean spellF1 by switch cost (descriptive):")
        print(" ".join(f"{k:g}:{v:.3f}" for k, v in grid.mean().items()))
        print("count ratio by cost: " + " ".join(f"{k:g}:{v:.2f}" for k, v in L.count_ratio.unstack("cost").mean().items()))
        print(f"learned* LOMO cost picks: {pd.Series(picks).value_counts().to_dict()}")
        print(f"VC TUNED   spellF1 {vc.f1.mean():.3f} (std {vc.f1.std():.3f})  count ratio {vc.count_ratio.mean():.2f}")
        print(f"learned*   spellF1 {star.f1.mean():.3f} (std {star.f1.std():.3f})  count ratio {star.count_ratio.mean():.2f}")
        diff = star.f1 - vc.f1
        print(f"paired learned* - VC: mean {diff.mean():+.3f}, min {diff.min():+.3f}, max {diff.max():+.3f}, "
              f"better in {(diff > 0).sum()}/{len(diff)}")


if __name__ == "__main__":
    main()
