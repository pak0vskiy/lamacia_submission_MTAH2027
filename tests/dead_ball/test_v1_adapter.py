import numpy as np

from dead_ball.idsse import Match
from dead_ball.pipeline import D, H
from dead_ball.v1_adapter import v1_codes, v1_frame

HL, HW, FPS = 52.5, 34.0, 25.0


def test_edge_sign(period_factory):
    f = v1_frame(period_factory(100), HL, HW, FPS)
    assert np.allclose(f.edge_m, HW)
    f = v1_frame(period_factory(100, ball=(0, HW + 1)), HL, HW, FPS)
    assert np.allclose(f.edge_m, -1)


def moving_period(period_factory, n):
    p = period_factory(n)
    drift = (np.arange(n) * 3 / FPS) % 20
    p.home[:, :, 0] += drift[:, None]
    p.away[:, :, 0] += drift[:, None]
    p.ball[:] = p.home[:, 5] + np.array([0.5, 0], np.float32)
    return p


def test_v1_runs(period_factory):
    p = moving_period(period_factory, 1500)
    m = Match("syn", FPS, HL, HW, [p])
    codes = v1_codes(p, m, None)
    assert len(codes) == 1500 and (codes == H).any()


def test_dead_score_substitution(period_factory):
    p = moving_period(period_factory, 1500)
    p.home[500:1000] = p.home[500:501]
    p.away[500:1000] = p.away[500:501]
    p.ball[500:1000] = p.ball[500]
    m = Match("syn", FPS, HL, HW, [p])
    score = np.zeros(1500)
    score[500:1000] = 1.0
    codes = v1_codes(p, m, score)
    assert (codes[600:1000] == D).any()
