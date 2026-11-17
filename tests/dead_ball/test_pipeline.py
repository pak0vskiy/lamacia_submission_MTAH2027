import numpy as np

from dead_ball.pipeline import H, concat, decode_dead, oracle_codes, score, truth_codes

FPS = 25.0


def runs_of(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(e[::2], e[1::2]))


def test_hysteresis():
    prob = np.full(500, 0.1)
    prob[100:175] = 0.9
    d = decode_dead(prob, FPS, enter=0.6, exit=0.4)
    assert runs_of(d) == [(100, 175)]


def test_min_dead():
    prob = np.full(500, 0.1)
    prob[100:125] = 0.9
    assert not decode_dead(prob, FPS, enter=0.6, exit=0.4).any()


def test_short_alive_filled():
    prob = np.full(500, 0.1)
    prob[100:175] = 0.9
    prob[187:262] = 0.9
    d = decode_dead(prob, FPS, enter=0.6, exit=0.4)
    assert runs_of(d) == [(100, 262)]


def test_oracle_fallback_next_owner(period_factory):
    dead = np.zeros(100, bool)
    dead[:10] = True
    p = period_factory(100, dead=dead, home_owns=False)
    p.home_owns[10:] = True
    codes = oracle_codes(p, np.zeros(100, bool))
    assert (codes[:10] == H).all()


def test_concat_no_straddle(period_factory):
    a = truth_codes(period_factory(300))
    gt = concat([a, a])
    r = score(gt, gt)
    assert r["n_gt_spells"] == 2 and r["spell_f1_mean"] == 1.0
