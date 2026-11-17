"""Run Sam's v1 (dead_ball/sam_v1.py, unmodified) on IDSSE periods, optionally with our dead score.

v1 expects 0..length / 0..width metres and ATD's per-player speed; IDSSE is centre-origin and we
derive speeds from positions (as we must for our ATD too).
"""

import warnings

import numpy as np
import pandas as pd

from dead_ball import sam_v1
from dead_ball.features import _slot_speed
from dead_ball.pipeline import A, D, H, X

# origin/sam_v1:lamacia_v1_config.json minus the match-specific keys (n_frames, length, width, halves)
V1_CONFIG = dict(motion_window_s=1.0, min_players=12, control_radius_m=3.0, margin_scale_m=1.5,
                 ownership_threshold=0.35, slow_speed_mps=0.9, fast_speed_mps=1.8,
                 stationary_speed_mps=0.6, edge_distance_m=2.0, dead_enter_threshold=0.5,
                 dead_exit_threshold=0.5, ownership_confirm_s=0.6, dead_enter_confirm_s=2.0,
                 dead_exit_confirm_s=0.6, max_motion_gap_s=0.12)
CODE = {"H": H, "A": A, "D": D, "X": X}


def v1_frame(p, half_len, half_wid, fps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spd = np.concatenate([_slot_speed(p.home, fps), _slot_speed(p.away, fps)], axis=1)
        bx = p.ball[:, 0] + half_len
        by = p.ball[:, 1] + half_wid
        vis = np.isfinite(bx)
        dist = lambda xy: np.nanmin(np.linalg.norm(xy - p.ball[:, None, :], axis=-1), axis=1)  # noqa: E731
        d_home, d_away = dist(p.home), dist(p.away)
        edge = np.minimum.reduce([bx, 2 * half_len - bx, by, 2 * half_wid - by])
        return pd.DataFrame(dict(
            median_speed=np.nanmedian(spd, axis=1), n_speed=np.isfinite(spd).sum(1),
            ball_x_m=bx, ball_y_m=by, ball_visible=vis.astype(float),
            d_home_m=np.where(vis, d_home, np.nan), d_away_m=np.where(vis, d_away, np.nan),
            edge_m=np.where(vis, edge, np.nan)))


def v1_codes(p, m, dead_score):
    """v1 labels as codes for one period. See v1_codes_from_frame."""
    return v1_codes_from_frame(v1_frame(p, m.half_len, m.half_wid, m.fps), m, dead_score)


def v1_codes_from_frame(f, m, dead_score):
    """v1 labels as codes. With dead_score (0..1 per frame), it replaces v1's dead evidence and
    1 - dead_score replaces its restart evidence; v1's owner evidence and decoder are untouched."""
    n = len(f)
    c = sam_v1.Config(fps=m.fps, n_frames=n, length=2 * m.half_len, width=2 * m.half_wid,
                      halves=((0, n - 1),), **V1_CONFIG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        owner, dead, live = sam_v1.evidence(f, c)
    if dead_score is not None:
        dead = np.asarray(dead_score, float)
        live = 1.0 - dead
    labels = sam_v1.decode(owner, dead, live, c)
    return np.array([CODE[k] for k in labels], np.int8)
