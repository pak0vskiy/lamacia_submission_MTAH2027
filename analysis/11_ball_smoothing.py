"""How much smoothing makes the degraded IDSSE ball move like our ATD ball? Label-free, aggregates only.

    uv run python analysis/11_ball_smoothing.py

Statistic: the frame-to-frame change of ball speed (m/s) on visible frames during open play.
Open play is a label-free proxy: players' mean speed over 1 s > 1.5 m/s, computed the same way on
both sides. On IDSSE (held-out test copies, c = 0.5) we apply dead_ball.degrade.smooth_ball with
a grid of Gaussian sigmas and pick the one whose q50/q90/q99 best match BAR-ATH (sum of
|log ratio|). BAR-ATH: graded frames, tracking only, no labels of any kind.
"""

import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dead_ball import cv  # noqa: E402
from dead_ball.degrade import smooth_ball  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402

FPS = 25.0
SIGMAS = (0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5)
QS = (0.5, 0.9, 0.99)
OPEN_PLAY_MPS = 1.5


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "analysis", file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def open_play(p):
    xy = np.concatenate([p.home, p.away], axis=1)
    lag = 5
    sp = np.full(xy.shape[:2], np.nan)
    sp[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=2) * FPS / lag
    m = pd.Series(np.nanmean(sp, axis=1)).rolling(int(FPS), center=True, min_periods=1).mean()
    return m.to_numpy() > OPEN_PLAY_MPS


def dspeed(ball, mask):
    sp = np.full(len(ball), np.nan)
    sp[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    d = np.abs(np.diff(sp))
    d = d[mask[1:]]
    return d[np.isfinite(d)]


def fmt(q):
    return " / ".join(f"{v:6.2f}" for v in q)


def main():
    warnings.simplefilter("ignore")
    a08 = _load("a08", "08_atd_vs_idsse_features.py")
    a09 = _load("a09", "09_team_possession_cv.py")
    atd, _, _ = a08.atd_periods()
    q_atd = np.quantile(np.concatenate([dspeed(p.ball, open_play(p)) for p in atd]), QS)
    open_share = np.mean(np.concatenate([open_play(p) for p in atd]))
    print(f"BAR-ATH open play {open_share:.1%} of graded frames; |d speed| q50/q90/q99 = {fmt(q_atd)}")

    ids = cv.default_ids()
    j = cv.TEST_COINCIDENCE.index(0.5)
    periods = []
    for i, mid in enumerate(ids):
        periods += a09.rebuild(load_match(mid), i, j, 0.5)
    masks = [open_play(p) for p in periods]
    print(f"IDSSE degraded (c = 0.5) open play {np.mean(np.concatenate(masks)):.1%} of frames\n")
    print(f"{'sigma s':>8}  q50 / q90 / q99           |log ratio| sum")
    best = None
    for s in SIGMAS:
        q = np.quantile(np.concatenate([dspeed(smooth_ball(p.ball, FPS, s), m)
                                        for p, m in zip(periods, masks)]), QS)
        err = float(np.abs(np.log(q / q_atd)).sum())
        print(f"{s:8.2f}  {fmt(q)}   {err:.3f}")
        if best is None or err < best[1]:
            best = (s, err)
    print(f"\nbest sigma = {best[0]:.2f} s (error {best[1]:.3f})")


if __name__ == "__main__":
    main()
