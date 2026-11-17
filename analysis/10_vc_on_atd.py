"""Does the Vidal-Codina team logic behave on our ATD like on degraded IDSSE? Aggregates only.

    uv run python analysis/10_vc_on_atd.py

There are no labels on BAR-ATH: no GT, public or private. We compare the rule's own behaviour on
our graded frames with the held-out degraded IDSSE test copies (c = 0.5), where truth is known:
- ball-speed noise (the rule fires on speed jumps > bv_threshold);
- how often the ball is in someone's zone, gains per live minute, and what triggered them;
- possessions per 90 min and their length; live time left X.
D on both sides comes from our dead-ball model (BAR-ATH: 7-fold mean; IDSSE: the match's fold).
Plausibility only. Nothing is tuned on BAR-ATH.
"""

import argparse
import importlib.util
from dataclasses import replace
import os
import sys
import warnings
from collections import Counter

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dead_ball import cv  # noqa: E402
from dead_ball.features import frame_features  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.pipeline import A, H, decode_dead, truth_codes  # noqa: E402
from dead_ball.degrade import smooth_ball  # noqa: E402
from possession.vidal_codina import VCParams, ball_speed, player_possession, team_codes  # noqa: E402

PARAMS = VCParams()          # set from the command line in main()
MODEL = "model-no-ball_spd_1@mixed"
FPS = 25.0
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "analysis", file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runs(codes):
    """Lengths (frames) of H/A runs."""
    ha = np.isin(codes, (H, A))
    e = np.flatnonzero(np.diff(np.r_[0, ha & (codes == H), 0]) != 0)
    out = list(e[1::2] - e[::2])
    e = np.flatnonzero(np.diff(np.r_[0, ha & (codes == A), 0]) != 0)
    return out + list(e[1::2] - e[::2])


def stats(periods, deads):
    s = Counter()
    jumps, lengths = [], []
    for p, dead in zip(periods, deads):
        players = np.concatenate([p.home, p.away], axis=1)
        team = np.r_[np.zeros(p.home.shape[1], int), np.ones(p.away.shape[1], int)]
        gains = []
        owner = player_possession(p.ball, players, FPS, PARAMS, gains=gains)
        codes = team_codes(owner, team, dead)
        live = ~dead
        vis = np.isfinite(p.ball[:, 0])
        sp = ball_speed(p.ball, FPS)
        dj = np.abs(np.diff(sp))
        dj = dj[np.isfinite(dj) & live[1:]]
        jumps.append(dj)
        dist = np.linalg.norm(players - p.ball[:, None, :], axis=2)
        near = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), axis=1) < PARAMS.pz_radius
        g_live = [g for g in gains if live[g[0]]]
        s["frames"] += len(p.ball)
        s["live"] += live.sum()
        s["live_vis"] += (live & vis).sum()
        s["live_near"] += (live & near).sum()
        s["gains"] += len(g_live)
        s["g_speed_only"] += sum(g[1] and not g[2] for g in g_live)
        s["g_angle_only"] += sum(g[2] and not g[1] for g in g_live)
        s["g_both"] += sum(g[1] and g[2] for g in g_live)
        s["live_x"] += (live & (codes == 3)).sum()
        s["live_h"] += (codes == H).sum()
        r = runs(codes)
        s["poss"] += len(r)
        lengths += r
    jumps = np.concatenate(jumps)
    return s, jumps, np.array(lengths) / FPS


def report(name, s, jumps, lengths, truth=None):
    live_min = s["live"] / FPS / 60
    g = max(s["gains"], 1)
    print(f"\n{name}")
    print(f"  live share {s['live'] / s['frames']:.1%}; ball visible in live {s['live_vis'] / s['live']:.1%}; "
          f"ball within {PARAMS.pz_radius} m of a player {s['live_near'] / s['live_vis']:.1%} of visible live")
    print(f"  |d speed| between frames (live, m/s) q50/q90/q99 = "
          f"{' / '.join(f'{v:.2f}' for v in np.quantile(jumps, (0.5, 0.9, 0.99)))}; "
          f"> {PARAMS.bv_threshold} m/s: {np.mean(jumps > PARAMS.bv_threshold):.2%}")
    print(f"  gains {s['gains']} = {s['gains'] / live_min:.1f} per live min; trigger speed-only "
          f"{s['g_speed_only'] / g:.0%}, angle-only {s['g_angle_only'] / g:.0%}, both {s['g_both'] / g:.0%}")
    per90 = s["poss"] / (s["frames"] / FPS / 60) * 90
    print(f"  possessions {s['poss']} ({per90:.0f} per 90 min); length s "
          f"{' / '.join(f'{v:.1f}' for v in np.quantile(lengths, QS))} (q10..q90); "
          f"live left X {s['live_x'] / s['live']:.1%}; home share of H/A {s['live_h'] / max(s['live'] - s['live_x'], 1):.1%}")
    if truth is not None:
        t90 = len(truth) / (s["frames"] / FPS / 60) * 90
        print(f"  truth possessions {len(truth)} ({t90:.0f} per 90 min); length s "
              f"{' / '.join(f'{v:.1f}' for v in np.quantile(truth, QS))}")


def main():
    global PARAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pz", type=float, default=2.5)   # possession.vidal_codina.TUNED
    ap.add_argument("--bv", type=float, default=5.0)
    ap.add_argument("--ba", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--ball-smooth", type=float, default=0.25, help="IDSSE ball sigma in s (analysis/11)")
    a = ap.parse_args()
    PARAMS = VCParams(pz_radius=a.pz, bv_threshold=a.bv, ba_threshold=a.ba, window_frames=a.window)
    print(f"{PARAMS}; IDSSE ball smoothing sigma {a.ball_smooth} s")
    warnings.simplefilter("ignore")
    ids = cv.default_ids()
    a08 = _load("a08", "08_atd_vs_idsse_features.py")
    a09 = _load("a09", "09_team_possession_cv.py")
    model = cv.Model(drop=cv.DEFAULT_DROP)
    # joblib = pickle: only our own fold models, written locally by dead_ball/cv.py
    folds = [joblib.load(os.path.join(cv.OUT, f"fold_{MODEL}_{m}.joblib")) for m in ids]

    # IDSSE, held-out test copies at c = 0.5
    data = cv.prepare(ids, 2, cv.DEFAULT_MOTION)
    j = cv.TEST_COINCIDENCE.index(0.5)
    periods, deads, truth = [], [], []
    for i, mid in enumerate(ids):
        m = load_match(mid)
        qs = [replace(q, ball=smooth_ball(q.ball, FPS, a.ball_smooth)) for q in a09.rebuild(m, i, j, 0.5)]
        for q, f in zip(qs, data[mid]["test"][0.5].feats):
            periods.append(q)
            deads.append(decode_dead(model.predict(folds[i]["fitted"], f), FPS, **folds[i]["params"]))
            truth += runs(truth_codes(q))
        del m
    report(f"IDSSE degraded (c = 0.5, ball sigma {a.ball_smooth} s), VC on predicted D", *stats(periods, deads), np.array(truth) / FPS)

    # BAR-ATH, graded frames, no labels
    atd, hl, hw = a08.atd_periods()
    decoder = dict(Counter(tuple(sorted(f["params"].items())) for f in folds).most_common(1)[0][0])
    deads = [decode_dead(np.mean([model.predict(f["fitted"], frame_features(p, FPS, hl, hw)) for f in folds],
                                 axis=0), FPS, **decoder) for p in atd]
    report("BAR-ATH graded frames, VC on predicted D (no labels; plausibility only)", *stats(atd, deads))


if __name__ == "__main__":
    main()
