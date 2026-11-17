import numpy as np

from dead_ball.features import frame_features

HL, HW, FPS = 52.5, 34.0, 25.0


def test_stationary_players_zero_speed(period_factory):
    f = frame_features(period_factory(500), FPS, HL, HW)
    mid = f.iloc[100:400]
    assert np.allclose(mid.spd_med_1, 0) and np.allclose(mid.still_share_3, 1)


def test_moving_players_speed(period_factory):
    p = period_factory(500)
    p.home[:, :, 0] += (np.arange(500) * 3 / FPS)[:, None]
    p.away[:, :, 0] += (np.arange(500) * 3 / FPS)[:, None]
    f = frame_features(p, FPS, HL, HW)
    assert np.allclose(f.spd_med_1.iloc[100:400], 3, atol=0.05)


def test_ball_edge_sign(period_factory):
    f = frame_features(period_factory(200), FPS, HL, HW)
    assert np.allclose(f.ball_edge, min(HL, HW))
    f = frame_features(period_factory(200, ball=(HL + 1, 0)), FPS, HL, HW)
    assert np.allclose(f.ball_edge, -1) and (f.ball_out_last == 1).all()


def test_time_since_ball(period_factory):
    p = period_factory(350)
    p.ball[100:] = np.nan
    f = frame_features(p, FPS, HL, HW)
    assert abs(f.t_since_ball.iloc[349] - 10.0) < 0.04
    assert f.t_until_ball.iloc[349] == 60


def test_never_visible_ball(period_factory):
    p = period_factory(300)
    p.ball[:] = np.nan
    f = frame_features(p, FPS, HL, HW)
    assert (f.ball_vis == 0).all()
    assert np.isfinite(f[["spd_med_1", "spread_home", "n_centre_circle"]].to_numpy()).all()
