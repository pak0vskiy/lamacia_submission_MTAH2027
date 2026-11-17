"""Metrica event recipe on synthetic events (no data files read)."""

import numpy as np
import pandas as pd

from dead_ball.metrica import labels

FPS = 25.0


def ev(rows):
    return pd.DataFrame(rows, columns=["period", "frame", "frame_end", "team", "type", "sub"])


def test_owner_ignores_challenges_and_switches_on_recovery():
    e = ev([(1, 100, 110, "home", "SET PIECE", "KICK OFF"),
            (1, 110, 150, "home", "PASS", ""),
            (1, 160, 165, "away", "CHALLENGE", "GROUND-WON"),   # duel: owner stays home
            (1, 200, 230, "home", "BALL LOST", "INTERCEPTION"),
            (1, 230, 231, "away", "RECOVERY", "INTERCEPTION"),
            (1, 240, 300, "away", "PASS", "")])
    frames = np.arange(100, 301)
    dead, home = labels(e, frames, 1, FPS)
    assert not dead.any()
    assert home[frames < 230].all()          # through the duel and the intercepted pass
    assert not home[frames >= 230].any()     # away from its recovery on


def test_dead_from_ball_out_to_restart_and_outside_events():
    e = ev([(1, 100, 110, "home", "SET PIECE", "KICK OFF"),
            (1, 110, 150, "home", "PASS", ""),
            (1, 150, 150, "home", "BALL OUT", ""),
            (1, 400, 410, "away", "SET PIECE", "THROW IN"),
            (1, 410, 450, "away", "PASS", "")])
    frames = np.arange(50, 500)
    dead, home = labels(e, frames, 1, FPS)
    assert dead[frames < 100].all() and dead[frames > 450].all()
    assert dead[(frames >= 150) & (frames < 400)].all()
    assert not dead[(frames >= 100) & (frames < 150)].any()
    assert not dead[(frames >= 400) & (frames <= 450)].any()
    assert not home[frames >= 400].any()
