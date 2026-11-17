import numpy as np

from dead_ball.degrade import GAP_QUANTILES, ball_mask, calibrated_mask, degrade, sample_gap_s


def test_gap_quantiles():
    g = sample_gap_s(np.random.default_rng(0), 20000)
    assert abs(np.median(g) - GAP_QUANTILES[2]) < 0.8
    assert abs(np.quantile(g, 0.9) - GAP_QUANTILES[4]) < 3


def test_missing_share():
    dead = np.zeros(100_000, bool)
    dead[::200] = True
    for c in (0.0, 0.9):
        m = ball_mask(dead, 25.0, np.random.default_rng(1), coincidence=c)
        assert abs(m.mean() - 0.41) < 0.02


def test_coincidence():
    dead = np.zeros(100_000, bool)
    dead[:50_000] = True
    hi = ball_mask(dead, 25.0, np.random.default_rng(2), coincidence=0.9)
    lo = ball_mask(dead, 25.0, np.random.default_rng(2), coincidence=0.0)
    assert dead[hi].mean() >= 0.8
    assert abs(dead[lo].mean() - 0.5) < 0.1


def test_clipped_at_edges():
    m = ball_mask(np.zeros(500, bool), 25.0, np.random.default_rng(3), coincidence=0.5)
    assert len(m) == 500


def test_degrade_keeps_labels(period_factory):
    p = period_factory(5000, dead=np.r_[np.zeros(2500, bool), np.ones(2500, bool)])
    q = degrade(p, 25.0, np.random.default_rng(4), coincidence=0.5)
    assert np.array_equal(q.dead, p.dead) and np.array_equal(q.t, p.t)
    assert np.array_equal(q.home_owns, p.home_owns)
    hidden = np.isnan(q.ball[:, 0])
    assert 0 < hidden.mean() < 1
    assert np.array_equal(np.isnan(q.ball[:, 1]), hidden)
    assert np.isfinite(p.ball).all()   # input untouched


def test_dead_ball_not_frozen(period_factory):
    dead = np.zeros(3000, bool)
    dead[1000:2000] = True
    p = period_factory(3000, dead=dead)
    p.ball[:1000] = np.array([10.0, 0.0], np.float32)
    p.ball[2000:] = np.array([30.0, 5.0], np.float32)
    p.ball[1000:2000] = p.ball[999]          # DFL-style freeze while dead
    q = degrade(p, 25.0, np.random.default_rng(5), coincidence=0.0)
    b = q.ball[1000:2000].astype(float)
    ok = np.isfinite(b[1:, 0]) & np.isfinite(b[:-1, 0])
    step = np.hypot(*np.diff(b, axis=0).T)[ok]
    assert (step == 0).mean() < 0.05
    # it travels toward the restart point
    vis = np.flatnonzero(np.isfinite(b[:, 0]))
    assert b[vis[-1], 0] > b[vis[0], 0]


def _exit_then_freeze(n_alive=500, n_dead=750, speed=10.0):
    """Ball runs +x at `speed` and crosses the line, then DFL-style freeze, then restart elsewhere."""
    n = 2 * n_alive + n_dead
    ball = np.zeros((n, 2), np.float32)
    ball[:n_alive, 0] = 52.5 - speed * (n_alive - np.arange(n_alive)) / 25.0
    ball[n_alive:n_alive + n_dead] = ball[n_alive - 1]
    ball[n_alive + n_dead:] = np.array([45.0, 20.0], np.float32)
    dead = np.zeros(n, bool)
    dead[n_alive:n_alive + n_dead] = True
    return ball, dead, n_alive, n_dead


def test_mixed_overshoots_past_line():
    from dead_ball.degrade import dead_ball_path
    ball, dead, s, _ = _exit_then_freeze()
    out = dead_ball_path(ball, dead, np.random.default_rng(6))
    assert out[s + 25, 0] - ball[s - 1, 0] > 1.0     # still travelling 1 s after the flag


def test_mixed_reaches_restart():
    from dead_ball.degrade import dead_ball_path
    ball, dead, s, nd = _exit_then_freeze()
    out = dead_ball_path(ball, dead, np.random.default_rng(7))
    assert np.hypot(*(out[s + nd - 1] - ball[s + nd])) < 1.0


def test_mixed_speed_mix():
    from dead_ball.degrade import dead_ball_path
    speeds = []
    for k in range(40):
        ball, dead, s, nd = _exit_then_freeze()
        out = dead_ball_path(ball, dead, np.random.default_rng(100 + k)).astype(float)
        v = np.hypot(*np.diff(out[s:s + nd], axis=0).T) * 25
        speeds.append(v)
    v = np.concatenate(speeds)
    assert (v < 0.5).mean() >= 0.2 and (v > 5).mean() >= 0.03


def test_smooth_ball_keeps_gaps_and_softens_kicks():
    from dead_ball.degrade import smooth_ball
    n = 200
    ball = np.zeros((n, 2), np.float32)
    ball[100:, 0] = (np.arange(n - 100) + 1) * 0.6        # kick: 0 → 15 m/s in one frame
    ball[150:160] = np.nan
    out = smooth_ball(ball, 25.0, sigma_s=0.2)
    assert np.array_equal(np.isnan(out[:, 0]), np.isnan(ball[:, 0]))
    jump = lambda b: np.nanmax(np.abs(np.diff(np.linalg.norm(np.diff(b, axis=0), axis=1) * 25)))
    assert jump(ball) > 10 and jump(out) < 5
    assert np.array_equal(smooth_ball(ball, 25.0, sigma_s=0.0), ball, equal_nan=True)


def test_calibrated_mask_rates(period_factory):
    # alternating 60 s live / 20 s dead spells, ~6 h of play
    unit = np.r_[np.zeros(1500, bool), np.ones(500, bool)]
    p = period_factory(len(unit) * 270, dead=np.tile(unit, 270))
    m = calibrated_mask(p, 25.0, np.random.default_rng(5))
    live, dead = ~p.dead, p.dead
    assert 0.6 < m[dead].mean() < 0.95          # stoppages mostly hidden
    assert m[live].mean() < 0.25                 # live play mostly visible
    assert len(m) == len(p.dead)


def test_calibrated_deterministic_and_labels_kept(period_factory):
    p = period_factory(5000, dead=np.r_[np.zeros(2500, bool), np.ones(2500, bool)])
    a = degrade(p, 25.0, np.random.default_rng(6), 0.5, blackout="calibrated")
    b = degrade(p, 25.0, np.random.default_rng(6), 0.5, blackout="calibrated")
    assert np.array_equal(np.isnan(a.ball[:, 0]), np.isnan(b.ball[:, 0]))
    assert np.array_equal(a.dead, p.dead)
