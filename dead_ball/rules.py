"""Hand rules for dead ball, shaped by the model read-out (analysis/07_dead_ball_patterns.py).

Read-out (7 fold models, held-out matches, c=0.5): the model relies on team mean speed over 3 s
(P(dead) crosses 0.5 near 1.45 m/s), ball speed (near 1.2 m/s), ball distance to the line (near
0.5 m) and whether players speed up in the next 3 s (near 1.5 m/s); a depth-4 surrogate on those
reproduces 94 % of its calls. Three rules, any one fires:

  R1 settled:   players slow AND not about to speed up AND (ball slow OR ball not seen)
  R2 ball out:  players slow AND ball at / over a line
  R3 placed:    ball slow AND ball near a line (players may still be moving into position)

Thresholds are fit on the training folds only (coordinate search on frame F1 of dead).
"""

import itertools

import numpy as np

DEFAULT_PARAMS = dict(team_slow=1.45, next_calm=1.5, ball_slow=1.2, edge_out=0.5, edge_near=5.6)
GRIDS = dict(team_slow=(1.0, 1.2, 1.45, 1.7, 2.0), next_calm=(1.0, 1.25, 1.5, 1.75, 2.0),
             ball_slow=(0.6, 0.9, 1.2, 1.5, 2.0), edge_out=(0.0, 0.5, 1.0, 2.0),
             edge_near=(2.0, 4.0, 5.6, 8.0))
FIT_ROWS = 200_000


def rule_score(F, params):
    team_slow = F.spd_mean_3.to_numpy() < params["team_slow"]
    calm = F.spd_med_next_3.to_numpy() < params["next_calm"]
    bspd = F.ball_spd_1.to_numpy()
    edge = F.ball_edge.to_numpy()
    ball_slow = np.isfinite(bspd) & (bspd < params["ball_slow"])
    ball_unseen = ~np.isfinite(bspd)
    out = np.isfinite(edge) & (edge < params["edge_out"])
    near = np.isfinite(edge) & (edge < params["edge_near"])
    r1 = team_slow & calm & (ball_slow | ball_unseen)
    r2 = team_slow & out
    r3 = ball_slow & near
    return (r1 | r2 | r3).astype(float)


def _f1(pred, y):
    tp = np.sum(pred & y)
    return 2 * tp / max(pred.sum() + y.sum(), 1)


def fit_rules(X, y, passes=2, seed=0):
    if len(X) > FIT_ROWS:
        idx = np.random.default_rng(seed).choice(len(X), FIT_ROWS, replace=False)
        X, y = X.iloc[idx], y[idx]
    params = dict(DEFAULT_PARAMS)
    for _, name in itertools.product(range(passes), GRIDS):
        scores = {v: _f1(rule_score(X, {**params, name: v}) > 0.5, y) for v in GRIDS[name]}
        params[name] = max(scores, key=scores.get)
    return params


class Rules:
    name = "rules"

    def fit(self, X, y):
        return fit_rules(X, y)

    def predict(self, fitted, X):
        return rule_score(X, fitted)
