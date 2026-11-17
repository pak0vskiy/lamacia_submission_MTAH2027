"""Per-frame windowed features from x, y only (what our ATD has). Centre-origin metres.

Offline problem, so centred and look-ahead windows are allowed.
"""

import warnings

import numpy as np
import pandas as pd

SPEED_LAG_S = 0.2
STILL_MPS = 0.5
CAP_S = 60.0
CORNER_R = 6.0
CENTRE_R = 9.15
BOX_DEPTH, BOX_HALF_W = 16.5, 20.16


def _roll(x, fps, seconds, how="mean", mode="centre"):
    w = max(1, int(round(seconds * fps)))
    s = pd.Series(x)
    if mode == "next":
        s = s[::-1]
    r = s.rolling(w, center=(mode == "centre"), min_periods=1)
    out = getattr(r, how)()
    return (out[::-1] if mode == "next" else out).to_numpy()


def _slot_speed(xy, fps):
    lag = max(1, int(round(SPEED_LAG_S * fps)))
    v = np.full(xy.shape[:2], np.nan, np.float32)
    v[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1) / (lag / fps)
    return v


def _time_since(visible, fps):
    n = len(visible)
    idx = np.where(visible, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, (np.arange(n) - last) / fps, CAP_S)
    return np.minimum(out, CAP_S)


def _gk_xy(xy, gk):
    out = np.full((len(xy), 2), np.nan, np.float32)
    ok = gk >= 0
    out[ok] = xy[np.flatnonzero(ok), gk[ok]]
    return out


def frame_features(p, fps, half_len, half_wid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return _features(p, fps, half_len, half_wid)


def _features(p, fps, hl, hw):
    allp = np.concatenate([p.home, p.away], axis=1)
    present = np.isfinite(allp[..., 0])
    n_present = present.sum(1)
    spd = np.concatenate([_slot_speed(p.home, fps), _slot_speed(p.away, fps)], axis=1)
    med = np.nanmedian(spd, axis=1)
    mean = np.nanmean(spd, axis=1)
    still = np.nansum(spd < STILL_MPS, axis=1) / np.maximum(np.isfinite(spd).sum(1), 1)
    still = np.where(np.isfinite(spd).any(1), still, np.nan)

    f = {}
    f["spd_med_1"] = _roll(med, fps, 1)
    f["spd_med_3"] = _roll(med, fps, 3)
    f["spd_mean_3"] = _roll(mean, fps, 3)
    f["spd_med_past_3"] = _roll(med, fps, 3, mode="past")
    f["spd_med_next_3"] = _roll(med, fps, 3, mode="next")
    f["spd_change_3"] = f["spd_med_next_3"] - f["spd_med_past_3"]
    f["still_share_3"] = _roll(still, fps, 3)
    f["n_players"] = n_present.astype(np.float32)

    for name, xy in (("home", p.home), ("away", p.away)):
        c = np.nanmean(xy, axis=1, keepdims=True)
        f[f"spread_{name}"] = _roll(np.nanmean(np.linalg.norm(xy - c, axis=-1), axis=1), fps, 1)
    f["hull_all"] = np.nanstd(allp[..., 0], axis=1) * np.nanstd(allp[..., 1], axis=1)

    bx, by = p.ball[:, 0], p.ball[:, 1]
    vis = np.isfinite(bx)
    f["ball_vis"] = vis.astype(np.float32)
    f["ball_vis_share_5"] = _roll(vis.astype(float), fps, 5)
    f["t_since_ball"] = _time_since(vis, fps)
    f["t_until_ball"] = _time_since(vis[::-1], fps)[::-1]
    bspd = np.full(len(bx), np.nan)
    bspd[1:] = np.hypot(np.diff(bx), np.diff(by)) * fps
    f["ball_spd_1"] = _roll(bspd, fps, 1)
    edge = np.minimum(hl - np.abs(bx), hw - np.abs(by))
    edge = pd.Series(edge).ffill().to_numpy()
    f["ball_edge"] = edge
    f["ball_out_last"] = np.where(np.isfinite(edge), (edge < 0).astype(float), np.nan)

    ax, ay = allp[..., 0], allp[..., 1]
    corner = np.zeros(len(ax))
    for sx in (-1, 1):
        for sy in (-1, 1):
            corner += np.nansum(np.hypot(ax - sx * hl, ay - sy * hw) < CORNER_R, axis=1)
    f["n_near_corner"] = corner
    in_box = [np.nansum((sx * ax > hl - BOX_DEPTH) & (np.abs(ay) < BOX_HALF_W), axis=1) for sx in (-1, 1)]
    f["n_in_box_max"] = np.maximum(*in_box).astype(float)
    f["n_centre_circle"] = np.nansum(np.hypot(ax, ay) < CENTRE_R, axis=1).astype(float)
    own = 0
    for xy in (p.home, p.away):
        side = np.sign(np.nanmean(xy[..., 0], axis=1, keepdims=True))
        own = own + np.nansum(np.sign(xy[..., 0]) == side, axis=1)
    f["own_half_share"] = own / np.maximum(n_present, 1)

    for name, xy, gk, sp in (("home", p.home, p.home_gk, _slot_speed(p.home, fps)),
                             ("away", p.away, p.away_gk, _slot_speed(p.away, fps))):
        g = _gk_xy(xy, gk)
        gs = np.full(len(g), np.nan)
        ok = gk >= 0
        gs[ok] = sp[np.flatnonzero(ok), gk[ok]]
        f[f"gk_spd_{name}"] = _roll(gs, fps, 1)
        f[f"gk_goal_dist_{name}"] = np.hypot(g[:, 0] - np.sign(g[:, 0]) * hl, g[:, 1])
    f["gk_goal_dist_min"] = np.fmin(f.pop("gk_goal_dist_home"), f.pop("gk_goal_dist_away"))

    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
