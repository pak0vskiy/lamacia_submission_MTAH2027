import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from possession.vidal_codina import (  # noqa: E402
    VCParams, ball_speed, player_possession, team_codes)
from dead_ball.pipeline import A, D, H, X  # noqa: E402

FPS = 25.0


def _pass(n=100, t_pass=40, t_arrive=60):
    """Player 0 (home) holds the ball at x=0, kicks it at t_pass, player 1 (away) at x=20 receives it."""
    players = np.zeros((n, 2, 2), np.float32)
    players[:, 1, 0] = 20.0
    ball = np.zeros((n, 2), np.float32)
    ball[:t_pass, 0] = 0.3
    ball[t_pass:t_arrive, 0] = np.linspace(0.3, 19.7, t_arrive - t_pass)
    ball[t_arrive:, 0] = 19.7
    return ball, players, np.array([0, 1])


def test_ball_speed_from_positions():
    ball = np.zeros((5, 2), np.float32)
    ball[:, 0] = np.arange(5) * 0.4          # 0.4 m / frame = 10 m/s
    ball[3] = np.nan
    s = ball_speed(ball, FPS)
    assert np.isclose(s[1], 10.0) and np.isnan(s[3]) and np.isnan(s[4])


def test_pass_gives_two_possessions():
    ball, players, _ = _pass()
    owner = player_possession(ball, players, FPS)
    assert owner[10] == 0          # holder: the kick is a speed jump inside his zone → valid gain
    assert owner[70] == 1          # receiver after the ball stops at his feet
    assert (owner[45:55] == -1).all()   # ball in flight: nobody


def test_gains_are_reported():
    ball, players, _ = _pass()
    gains = []
    player_possession(ball, players, FPS, gains=gains)
    assert len(gains) == 2 and all(g[1] for g in gains)   # kick and stop are both speed jumps


def test_no_gain_without_speed_or_angle_change():
    ball, players, _ = _pass()
    # Ball rolls past player 1 at constant velocity: in zone, but no change of speed or direction.
    ball[:] = np.nan
    ball[:, 0] = np.linspace(10, 30, len(ball))
    ball[:, 1] = 0.5
    owner = player_possession(ball, players, FPS, VCParams(bv_threshold=5.0, ba_threshold=10.0))
    assert (owner == -1).all()


def test_team_codes_hold_through_flight_and_reset_at_dead():
    owner = np.full(100, -1)
    owner[10:20] = 0
    owner[50:60] = 1
    team = np.array([0, 1])
    dead = np.zeros(100, bool)
    dead[70:80] = True
    codes = team_codes(owner, team, dead)
    assert (codes[:10] == H).all()           # back-filled to interval start
    assert (codes[20:50] == H).all()         # held through the flight
    assert (codes[50:70] == A).all()
    assert (codes[70:80] == D).all()
    assert (codes[80:] == X).all()           # no gain after the stoppage: unknown, never carried over D


def test_team_codes_missing_ball_holds_owner():
    owner = np.full(50, -1)
    owner[:5] = 1
    codes = team_codes(owner, np.array([0, 1]), np.zeros(50, bool))
    assert (codes == A).all()


def test_window_trigger_survives_smoothing():
    from dead_ball.degrade import smooth_ball
    ball, players, _ = _pass()
    smooth = smooth_ball(ball, FPS, sigma_s=0.2)
    frame = player_possession(smooth, players, FPS, VCParams(ba_threshold=180.0))   # speed trigger only
    window = player_possession(smooth, players, FPS, VCParams(ba_threshold=180.0, window_frames=6))
    assert (frame[:30] == -1).all()            # frame-to-frame jump no longer reaches 5 m/s
    assert window[10] == 0 and window[90] == 1  # speed change across the zone ± 6 frames does


def test_window_one_is_the_paper_rule():
    ball, players, _ = _pass()
    assert np.array_equal(player_possession(ball, players, FPS),
                          player_possession(ball, players, FPS, VCParams(window_frames=1)))
