"""Kickoff detector on synthetic formations (no data files read)."""

import numpy as np

from dead_ball import kickoff

FPS = 25.0


def teams(n, rng, home_side):
    """home_side: (n,) -1 / +1 = home in the left / right half; 0 = mixed (live play)."""
    home = rng.uniform(5, 50, (n, 11, 2)) * [1, 0.6]
    away = rng.uniform(5, 50, (n, 11, 2)) * [1, 0.6]
    side = np.asarray(home_side, float)[:, None]
    mixed = side == 0
    home[..., 0] = np.where(mixed, rng.uniform(-50, 50, (n, 11)), home[..., 0] * side)
    away[..., 0] = np.where(mixed, rng.uniform(-50, 50, (n, 11)), -away[..., 0] * side)
    return home.astype(np.float32), away.astype(np.float32)


def test_detects_the_break_after_a_formation():
    rng = np.random.default_rng(0)
    walk, form, live = int(30 * FPS), int(6 * FPS), int(60 * FPS)
    side = np.r_[np.zeros(walk), -np.ones(form), np.zeros(live)]
    home, away = teams(len(side), rng, side)
    k = kickoff.detect(home, away, FPS, lag_s=0.0)
    assert k == walk + form
    assert kickoff.detect(home, away, FPS) == walk + form - int(round(kickoff.LAG_S * FPS))


def test_tolerates_one_outlier_and_ignores_short_or_unbroken_formations():
    rng = np.random.default_rng(1)
    side = np.r_[np.zeros(250), -np.ones(150), np.zeros(1500)]
    home, away = teams(len(side), rng, side)
    home[:, 0, 0] = 40.0                       # one home track always in the wrong half
    assert kickoff.detect(home, away, FPS, lag_s=0.0) == 400
    short = np.r_[np.zeros(250), -np.ones(40), np.zeros(1500)]     # 1.6 s: too short
    assert kickoff.detect(*teams(len(short), rng, short), FPS) is None
    unbroken = -np.ones(1000)                  # never breaks within the window
    assert kickoff.detect(*teams(len(unbroken), rng, unbroken), FPS) is None
