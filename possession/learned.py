"""Learned team logic: per-frame P(home has the ball) + switch-penalised decoding.

1. `team_features`: positions-only features for one period (centre-origin metres), offline, so
   they look both backwards and forwards in time (who reaches the ball next, where the team moves).
   Vidal-Codina's own calls (zone runs, accepted gains) go in as features too. Every team feature
   comes as a home_/away_ pair defined from that team's point of view, so swapping the teams
   swaps the columns exactly.
2. `fit` / `p_home`: gradient-boosted trees on live frames of degraded IDSSE (other games), trained
   on team-swapped duplicates and symmetrised at inference: p(swapped) = 1 - p.
3. `decode`: inside each predicted live stretch, a 2-state Viterbi with a fixed cost per team
   switch, the cost being the decoder's one parameter (chosen on other games only).

No labels, dead flags or owner fields are read by `team_features`.
"""

import numpy as np
import pandas as pd

from dead_ball.pipeline import A, D, H
from possession.vidal_codina import TUNED, ball_speed, player_possession

CAP_S = 10.0
SEED = 0
# Switch cost of the decoder: picked leave-one-match-out on IDSSE, 7/7 folds
# (observations/2026-11-09_learned-team-logic.md).
SWITCH_COST = 6.0
BALL_SMOOTH_S = 0.25          # IDSSE ball smoothed like our ATD's (analysis/11)
SUBSAMPLE = 5
N_COPIES = 2


def training_rows(m, i, n_copies=N_COPIES, subsample=SUBSAMPLE, ball_smooth=BALL_SMOOTH_S):
    """(X, y_home) from live frames of IDSSE match `m` (index i), degraded exactly like the
    dead-ball training copies (dead_ball.cv.prepare draws), ball smoothed like ATD."""
    from dataclasses import replace
    from dead_ball import cv
    from dead_ball.degrade import degrade, smooth_ball
    Xs, ys = [], []
    for k in range(n_copies):
        rng = np.random.default_rng([cv.SEED, i, k])
        c = rng.uniform(*cv.TRAIN_COINCIDENCE)
        for p in m.periods:
            q = degrade(p, m.fps, rng, c, motion=cv.DEFAULT_MOTION, blackout=cv.DEFAULT_BLACKOUT)
            q = replace(q, ball=smooth_ball(q.ball, m.fps, ball_smooth))
            F = team_features(q.ball, q.home, q.away, m.fps, m.half_len, m.half_wid)
            idx = np.arange(0, len(q.dead), subsample)
            idx = idx[~q.dead[idx]]
            Xs.append(F.iloc[idx])
            ys.append(q.home_owns[idx].astype(np.int8))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


# ---------------------------------------------------------------- small helpers

def _roll(x, w, how, side):
    """Rolling stat over w frames: side 'past' (trailing, incl. now), 'future' (leading), 'centre'."""
    s = pd.Series(np.asarray(x, np.float64))
    if side == "future":
        s = s[::-1].reset_index(drop=True)
    r = s.rolling(max(int(w), 1), min_periods=1, center=(side == "centre"))
    out = getattr(r, how)().to_numpy()
    return out[::-1].copy() if side == "future" else out


def _since(event, fps, cap=CAP_S):
    """Seconds since the last True of `event` (inclusive), capped."""
    i = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, i, -1))
    return np.where(last >= 0, np.minimum((i - last) / fps, cap), cap)


def _until(event, fps, cap=CAP_S):
    return _since(np.asarray(event)[::-1], fps, cap)[::-1]


def _team_speed_vx(xy, fps, lag=5):
    v = np.full(xy.shape, np.nan, np.float32)
    v[lag:] = (xy[lag:] - xy[:-lag]) * (fps / lag)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.linalg.norm(v, axis=2), axis=1), np.nanmean(v[:, :, 0], axis=1)


# ---------------------------------------------------------------- features

def team_features(ball, home, away, fps, half_len, half_wid, params=TUNED):
    """DataFrame of per-frame features. `ball` (n,2) may contain NaN; home/away (n,P,2)."""
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    n = len(ball)
    vis = np.isfinite(ball).all(axis=1)
    f = {
        "ball_visible": vis.astype(np.float32),
        "since_ball": _since(vis, fps),
        "until_ball": _until(vis, fps),
        "ball_edge": np.minimum(half_len - np.abs(ball[:, 0]), half_wid - np.abs(ball[:, 1])),
        "ball_speed": _roll(ball_speed(ball, fps), 5, "mean", "centre"),
        "ball_abs_y": np.abs(ball[:, 1]) / half_wid,
    }

    # Vidal-Codina's calls: who is in an accepted zone run, and where its gains start.
    players = np.concatenate([home, away], axis=1)
    team_of = np.r_[np.zeros(home.shape[1], int), np.ones(away.shape[1], int)]
    gains = []
    owner = player_possession(ball, players, fps, params, gains)
    owner_team = np.where(owner >= 0, team_of[np.maximum(owner, 0)], -1)
    gain_team = np.full(n, -1)
    for s, _, _ in gains:
        gain_team[s] = owner_team[s]

    hc = np.nanmean(home, axis=1)
    ac = np.nanmean(away, axis=1)
    # +1 if home's opponents sit at larger x (home attacks +x), smoothed over 10 s.
    sign = np.sign(_roll(np.nan_to_num(ac[:, 0] - hc[:, 0]), 10 * fps, "mean", "centre"))
    sign[sign == 0] = 1.0
    bvx = np.r_[np.nan, np.diff(ball[:, 0])] * fps

    for side, xy, c, own, d in (("home", home, hc, 0, sign), ("away", away, ac, 1, -sign)):
        dist = np.linalg.norm(xy - ball[:, None, :], axis=2)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        srt = np.sort(dist, axis=1)
        d1 = np.where(np.isfinite(srt[:, 0]), srt[:, 0], np.nan)
        d2 = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan) if xy.shape[1] > 1 else np.full(n, np.nan)
        near = (dist < 3.0).sum(axis=1).astype(np.float32)
        near[~vis] = np.nan
        spd, vx = _team_speed_vx(xy, fps)
        zone = (owner_team == own).astype(np.float32)
        g = gain_team == own
        t = {
            "d1": d1, "d2": d2, "near3": near,
            "d1_past05": _roll(d1, 0.5 * fps, "min", "past"), "d1_fut05": _roll(d1, 0.5 * fps, "min", "future"),
            "d1_past2": _roll(d1, 2 * fps, "min", "past"), "d1_fut2": _roll(d1, 2 * fps, "min", "future"),
            "last_d1": pd.Series(d1).ffill(limit=int(2 * fps)).to_numpy(),
            "next_d1": pd.Series(d1).bfill(limit=int(2 * fps)).to_numpy(),
            "zone_past1": _roll(zone, fps, "mean", "past"), "zone_fut1": _roll(zone, fps, "mean", "future"),
            "zone_past3": _roll(zone, 3 * fps, "mean", "past"), "zone_fut3": _roll(zone, 3 * fps, "mean", "future"),
            "since_gain": _since(g, fps), "until_gain": _until(g, fps),
            "speed1": _roll(spd, fps, "mean", "centre"),
            "adv_v1": _roll(vx * d, fps, "mean", "centre"),
            "adv_v3_fut": _roll(vx * d, 3 * fps, "mean", "future"),
            "adv_v3_past": _roll(vx * d, 3 * fps, "mean", "past"),
            "depth": c[:, 0] * d / half_len,
            "ball_rel_depth": (ball[:, 0] - c[:, 0]) * d,
            "ball_adv_v_past": _roll(bvx * d, 0.5 * fps, "mean", "past"),
            "ball_adv_v_fut": _roll(bvx * d, 0.5 * fps, "mean", "future"),
            "spread": np.nanmean(np.linalg.norm(xy - c[:, None, :], axis=2), axis=1),
            "length": np.nanstd(xy[:, :, 0], axis=1),
            "width": np.nanstd(xy[:, :, 1], axis=1),
        }
        for k, v in t.items():
            f[f"{side}_{k}"] = v
    out = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
    return out.replace([np.inf, -np.inf], np.nan)


def swap_teams(X):
    out = X.copy()
    for c in X.columns:
        if c.startswith("home_"):
            o = "away_" + c[5:]
            out[c], out[o] = X[o].to_numpy(), X[c].to_numpy()
    return out


# ---------------------------------------------------------------- model

def fit(X, y_home, seed=SEED):
    """y_home: 1 where home owns. Trained on X plus its team-swapped copy with flipped targets."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xa = pd.concat([X, swap_teams(X)], ignore_index=True)
    ya = np.r_[y_home, 1 - y_home].astype(np.int8)
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False)
    return clf.fit(Xa, ya)


def p_home(clf, X):
    j = list(clf.classes_).index(1)
    return 0.5 * (clf.predict_proba(X)[:, j] + 1.0 - clf.predict_proba(swap_teams(X))[:, j])


# ---------------------------------------------------------------- decoding

def viterbi2(p, switch_cost, eps=1e-4, start=None):
    """Most likely H/A path (True = home) for P(home) per frame, cost `switch_cost` per switch.
    start: None, or True / False to force the first frame to home / away."""
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    ch, ca = (-np.log(p)).tolist(), (-np.log(1 - p)).tolist()   # cost of home / away per frame
    if start is not None:
        ch[0], ca[0] = (0.0, np.inf) if start else (np.inf, 0.0)
    n = len(p)
    bh, ba = bytearray(n), bytearray(n)                          # 1 = came from the other state
    h, a = ch[0], ca[0]
    for t in range(1, n):
        hs, as_ = a + switch_cost, h + switch_cost
        if hs < h:
            bh[t] = 1
            nh = hs
        else:
            nh = h
        if as_ < a:
            ba[t] = 1
            na = as_
        else:
            na = a
        h, a = nh + ch[t], na + ca[t]
    home = np.zeros(n, bool)
    s = 0 if h <= a else 1
    for t in range(n - 1, -1, -1):
        home[t] = s == 0
        if (bh if s == 0 else ba)[t]:
            s = 1 - s
    return home


def live_stretches(dead):
    """[start, end) of each live stretch."""
    live = ~np.asarray(dead, bool)
    e = np.flatnonzero(np.diff(np.r_[0, live.astype(np.int8), 0]))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


def restart_starts(dead, restarts, window):
    """{live-stretch start: home?} from restart hints [(lo, hi, home)] (frames). A stretch takes
    the hint whose [lo - window, hi + window] contains its first frame (the nearest if several)."""
    out = {}
    for s, _ in live_stretches(dead):
        best = None
        for lo, hi, home in restarts:
            if lo - window <= s <= hi + window:
                d = 0 if lo <= s <= hi else min(abs(s - lo), abs(s - hi))
                if best is None or d < best[0]:
                    best = (d, home)
        if best is not None:
            out[s] = best[1]
    return out


def decode(p, dead, switch_cost, starts=None):
    """H/A/D codes: D where `dead`, else the Viterbi path inside each live stretch.
    starts: optional {stretch start: home?} forcing the team at the start of that stretch."""
    codes = np.full(len(p), D, np.int8)
    starts = starts or {}
    for s, t in live_stretches(dead):
        q = np.nan_to_num(p[s:t], nan=0.5)
        codes[s:t] = np.where(viterbi2(q, switch_cost, start=starts.get(s)), H, A)
    return codes
