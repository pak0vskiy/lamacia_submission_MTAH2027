import numpy as np
import pandas as pd

from dead_ball.rules import DEFAULT_PARAMS, rule_score


def frame(team_speed, ball_speed, edge, next_speed, n=50):
    return pd.DataFrame(dict(spd_mean_3=np.full(n, team_speed), ball_spd_1=np.full(n, ball_speed),
                             ball_edge=np.full(n, edge), spd_med_next_3=np.full(n, next_speed)))


def test_range():
    rng = np.random.default_rng(0)
    F = pd.DataFrame(dict(spd_mean_3=rng.uniform(0, 5, 500), ball_spd_1=rng.uniform(0, 20, 500),
                          ball_edge=rng.uniform(-2, 30, 500), spd_med_next_3=rng.uniform(0, 5, 500)))
    F.loc[::7, "ball_spd_1"] = np.nan
    s = rule_score(F, DEFAULT_PARAMS)
    assert ((s >= 0) & (s <= 1)).all()


def test_still_players_ball_out_is_dead():
    assert (rule_score(frame(0.3, 0.2, -1.0, 0.3), DEFAULT_PARAMS) == 1).all()


def test_fast_play_central_ball_is_alive():
    assert (rule_score(frame(4.0, 12.0, 20.0, 4.0), DEFAULT_PARAMS) == 0).all()


def test_missing_ball_slow_players_is_dead():
    assert (rule_score(frame(0.3, np.nan, np.nan, 0.3), DEFAULT_PARAMS) == 1).all()
