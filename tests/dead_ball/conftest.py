import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dead_ball.idsse import Period  # noqa: E402


def make_period(n, n_players=11, dead=None, ball=(0.0, 0.0), home_owns=True):
    """Synthetic period: players on a grid, ball fixed, all alive unless `dead` given."""
    xs = np.linspace(-40, 40, n_players, dtype=np.float32)
    home = np.zeros((n, n_players, 2), np.float32)
    home[:, :, 0] = xs
    home[:, :, 1] = -10
    away = home.copy()
    away[:, :, 1] = 10
    return Period(
        t=np.arange(n) / 25.0,
        dead=np.zeros(n, bool) if dead is None else dead,
        home_owns=np.full(n, home_owns),
        ball=np.tile(np.array(ball, np.float32), (n, 1)),
        home=home, away=away,
        home_gk=np.zeros(n, np.int16), away_gk=np.zeros(n, np.int16))


@pytest.fixture
def period_factory():
    return make_period
