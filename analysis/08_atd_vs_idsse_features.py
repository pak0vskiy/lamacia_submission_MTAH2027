"""Do the dead-ball model's inputs look the same on our ATD as on degraded IDSSE? Aggregates only.

    uv run python analysis/08_atd_vs_idsse_features.py

BAR-ATH side: graded frames of the tactical ATD, no labels of any kind (no GT, public or private).
IDSSE side: the held-out test copies at coincidence 0.5 from analysis/06 (degraded), split by the
DFL flag so we can see what the model learned to expect. "Slow play" (team mean speed over 3 s
< 1.0 m/s) is a label-free proxy for stoppages that exists on both sides.
Also reports the model's own predictions on BAR-ATH (dead share, stoppage count and lengths), as a
plausibility check only: nothing is tuned on it.
"""

import argparse
import os
import sys
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.atd_cache import graded_periods  # noqa: E402
from dead_ball import cv  # noqa: E402
from dead_ball.features import frame_features  # noqa: E402
from dead_ball.pipeline import decode_dead  # noqa: E402

FEATS = ["ball_spd_1", "spd_mean_3", "spd_med_next_3", "ball_edge", "still_share_3", "ball_vis",
         "t_since_ball"]
SLOW_PLAY = 1.0
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def atd_periods():
    return graded_periods()


def q(v):
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:6.2f}" for x in np.quantile(v, QS)) if len(v) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-no-ball_spd_1@mixed", help="fold model name from analysis/06")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--motion", default=cv.DEFAULT_MOTION, choices=["carry", "mixed"])
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    periods, hl, hw = atd_periods()
    atd = pd.concat([frame_features(p, 25.0, hl, hw) for p in periods], ignore_index=True)
    bx = np.concatenate([p.ball[:, 0] for p in periods]) + hl
    print(f"BAR-ATH graded frames {len(atd)}; ball x / length q01/q99 = "
          f"{np.nanquantile(bx, .01) / (2 * hl):.2f} / {np.nanquantile(bx, .99) / (2 * hl):.2f} "
          f"(sanity: coordinates span the pitch)")

    data = cv.prepare(cv.default_ids(), a.copies, a.motion)
    ids = pd.concat([f for d in data.values() for f in d["test"][0.5].feats], ignore_index=True)
    dead = np.concatenate([p.dead for d in data.values() for p in d["test"][0.5].periods])

    groups = {
        "BAR-ATH all": atd,
        "IDSSE all": ids,
        "IDSSE dead": ids[dead],
        "IDSSE alive": ids[~dead],
        "BAR-ATH slow play": atd[atd.spd_mean_3 < SLOW_PLAY],
        "IDSSE slow play": ids[ids.spd_mean_3 < SLOW_PLAY],
    }
    print(f"\nFeature quantiles q10 / q25 / q50 / q75 / q90 (slow play = spd_mean_3 < {SLOW_PLAY} m/s)")
    for feat in FEATS:
        print(f"\n  {feat}")
        for name, g in groups.items():
            print(f"    {name:<18} {q(g[feat].to_numpy(float))}")

    print("\nShares")
    for name, g in groups.items():
        vis = g.ball_vis.to_numpy() > 0
        b = g.ball_spd_1.to_numpy(float)
        print(f"  {name:<18} frames {len(g):8d}  ball visible {vis.mean():5.1%}  "
              f"ball_spd_1 < 1.2 m/s (of visible) {np.nanmean(b[vis] < 1.2):5.1%}  "
              f"slow-play share {np.mean(g.spd_mean_3 < SLOW_PLAY):5.1%}")

    # The model's own calls on BAR-ATH: mean of the 7 fold models (plausibility only).
    folds = [joblib.load(os.path.join(cv.OUT, f"fold_{a.model}_{m}.joblib"))   # written by cv.py
             for m in cv.default_ids()]
    method = cv.Model(drop=[c for c in ids.columns if c not in folds[0]["fitted"].feature_names_in_])
    decoder = Counter(tuple(sorted(f["params"].items())) for f in folds).most_common(1)[0][0]
    decoder = dict(decoder)
    print(f"\nModel '{a.model}', decoder (most common inner-CV choice) {decoder}")
    shares, lengths, n_spells = [], [], 0
    for p in periods:
        f = frame_features(p, 25.0, hl, hw)
        prob = np.mean([method.predict(fd["fitted"], f) for fd in folds], axis=0)
        d = decode_dead(prob, 25.0, **decoder)
        shares.append(d)
        e = np.flatnonzero(np.diff(np.r_[0, d.astype(np.int8), 0]))
        lengths += list((e[1::2] - e[::2]) / 25.0)
        n_spells += len(e) // 2
    true_len = []
    for dd in data.values():
        for p in dd["test"][0.5].periods:
            e = np.flatnonzero(np.diff(np.r_[0, p.dead.astype(np.int8), 0]))
            true_len += list((e[1::2] - e[::2]) / 25.0)
    graded_min = sum(len(p) for p in periods) / 25 / 60
    idsse_min = len(ids) / 25 / 60
    print("\nModel on BAR-ATH (no labels; plausibility only) vs IDSSE truth")
    print(f"  BAR-ATH predicted: dead share {np.concatenate(shares).mean():.1%}, stoppages {n_spells} "
          f"({n_spells / graded_min * 90:.0f} per 90 min), length s {q(np.array(lengths))}")
    print(f"  IDSSE truth:       dead share {dead.mean():.1%}, stoppages {len(true_len)} "
          f"({len(true_len) / idsse_min * 90:.0f} per 90 min), length s {q(np.array(true_len))}")


if __name__ == "__main__":
    main()
