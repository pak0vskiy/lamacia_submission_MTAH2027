import numpy as np
import pandas as pd

from dead_ball.pipeline import A, D, H
from possession import learned


def _scene(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ball = np.c_[np.linspace(-20, 20, n), 3 * np.sin(t / 15)].astype(np.float32)
    ball[50:70] = np.nan
    home = (ball[:, None, :] + rng.normal(0, 4, (n, 5, 2))).astype(np.float32)
    away = (ball[:, None, :] + rng.normal(0, 6, (n, 5, 2)) + [5, 0]).astype(np.float32)
    home[50:70] = np.nan_to_num(home[50:70], nan=0.0) + rng.normal(0, 1, (20, 5, 2))
    away[50:70] = np.nan_to_num(away[50:70], nan=5.0) + rng.normal(0, 1, (20, 5, 2))
    return ball, home, away


def test_features_swap_exactly_with_teams():
    ball, home, away = _scene()
    F = learned.team_features(ball, home, away, 25.0, 52.5, 34.0)
    G = learned.team_features(ball, away, home, 25.0, 52.5, 34.0)
    assert list(F.columns) == list(G.columns)
    assert np.allclose(G.to_numpy(), learned.swap_teams(F).to_numpy(), equal_nan=True, atol=1e-4)
    assert not any(k in c for c in F.columns for k in ("dead", "owns"))


def test_p_home_is_team_symmetric():
    ball, home, away = _scene()
    F = learned.team_features(ball, home, away, 25.0, 52.5, 34.0)
    y = (np.arange(len(F)) < 100).astype(np.int8)
    clf = learned.fit(F, y)
    p = learned.p_home(clf, F)
    q = learned.p_home(clf, learned.swap_teams(F))
    assert np.allclose(p + q, 1.0)


def test_viterbi_switch_cost():
    p = np.r_[np.full(50, 0.9), np.full(3, 0.1), np.full(50, 0.9), np.full(50, 0.2)]
    assert learned.viterbi2(p, 0.0)[50:53].sum() == 0          # free switching follows the blip
    path = learned.viterbi2(p, 8.0)
    assert path[:103].all() and not path[103:].any()           # blip smoothed, real switch kept


def test_decode_keeps_dead_and_resets():
    p = np.r_[np.full(20, 0.9), np.full(10, 0.5), np.full(20, 0.1)]
    dead = np.zeros(50, bool)
    dead[20:30] = True
    c = learned.decode(p, dead, 4.0)
    assert (c[20:30] == D).all() and (c[:20] == H).all() and (c[30:] == A).all()


def test_forced_start_and_restart_hints():
    p = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    assert learned.viterbi2(p, 6.0)[0]
    assert not learned.viterbi2(p, 6.0, start=False)[0]
    dead = np.r_[np.ones(10, bool), np.zeros(40, bool), np.ones(10, bool), np.zeros(40, bool)]
    assert learned.live_stretches(dead) == [(10, 50), (60, 100)]
    starts = learned.restart_starts(dead, [(5, 8, False), (70, 80, True)], window=3)
    assert starts == {10: False}                     # 60 is 10 frames before (70, 80): too far
    codes = learned.decode(np.full(100, 0.9), dead, 6.0, starts)
    assert codes[10] == learned.A and codes[60] == learned.H
