"""What did the dead-ball model learn? Aggregates only (importances, thresholds, a small tree).

    uv run python analysis/07_dead_ball_patterns.py [--copies K]

Needs analysis/06_dead_ball_cv.py to have run (fold models in artifacts/dead_ball/). Uses each
fold's held-out match at coincidence 0.5. Other games only.
"""

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dead_ball import cv  # noqa: E402

N_SAMPLE = 20_000
TOP_PD = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--model", default="model-no-ball_spd_1@mixed", help="fold model name from analysis/06")
    ap.add_argument("--motion", default=cv.DEFAULT_MOTION, choices=["carry", "mixed"])
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    data = cv.prepare(cv.default_ids(), a.copies, a.motion)
    rng = np.random.default_rng(cv.SEED)

    ranks, imps, pooled_X, pooled_p, models = [], [], [], [], {}
    for held, d in data.items():
        fold = joblib.load(os.path.join(cv.OUT, f"fold_{a.model}_{held}.joblib"))   # written by cv.py
        model = models[held] = fold["fitted"]
        copy = d["test"][0.5]
        X = pd.concat(copy.feats, ignore_index=True)[list(model.feature_names_in_)]
        y = np.concatenate([p.dead for p in copy.periods])
        idx = rng.choice(len(X), min(N_SAMPLE, len(X)), replace=False)
        pi = permutation_importance(model, X.iloc[idx], y[idx], n_repeats=3, random_state=cv.SEED,
                                    scoring="average_precision")
        s = pd.Series(pi.importances_mean, index=X.columns)
        imps.append(s)
        ranks.append(s.rank(ascending=False))
        pooled_X.append(X.iloc[idx])
        pooled_p.append(model.predict_proba(X.iloc[idx])[:, 1])

    imp = pd.DataFrame(imps)
    rank = pd.DataFrame(ranks)
    table = pd.DataFrame(dict(mean_rank=rank.mean(), rank_std=rank.std(),
                              importance=imp.mean(), imp_std=imp.std())).sort_values("mean_rank")
    print("Permutation importance (drop in average precision), held-out match, c=0.5")
    print(table.head(12).round(3).to_string())

    Xp = pd.concat(pooled_X, ignore_index=True)
    pp = np.concatenate(pooled_p)
    print("\nPartial dependence of P(dead), mean over fold models; grid = feature quantiles 5..95%")
    for feat in table.index[:TOP_PD]:
        curves = []
        sample = Xp[np.isfinite(Xp[feat])].sample(2000, random_state=cv.SEED).astype(np.float64)
        for m in models.values():
            r = partial_dependence(m, sample, [feat], method="brute",
                                   grid_resolution=20, percentiles=(0.05, 0.95), kind="average",
                                   response_method="predict_proba")
            curves.append(r["average"][0])
        grid = r["grid_values"][0]
        curve = np.mean(curves, axis=0)
        cross = np.flatnonzero(np.diff(np.sign(curve - 0.5)))
        where = ", ".join(f"{(grid[i] + grid[i + 1]) / 2:.2f}" for i in cross) or "none"
        trend = "rises" if curve[-1] > curve[0] else "falls"
        print(f"  {feat:<18} P(dead) {curve[0]:.2f} -> {curve[-1]:.2f} ({trend}) over "
              f"[{grid[0]:.2f}, {grid[-1]:.2f}]; crosses 0.5 at {where}")

    target = pp > 0.5
    tree = DecisionTreeClassifier(max_depth=4, random_state=cv.SEED, min_samples_leaf=200).fit(Xp, target)
    print(f"\nSurrogate tree (depth 4) fit to the model's calls: fidelity {tree.score(Xp, target):.1%}, "
          f"pooled {len(Xp)} held-out frames")
    print(export_text(tree, feature_names=list(Xp.columns), decimals=2))


if __name__ == "__main__":
    main()
