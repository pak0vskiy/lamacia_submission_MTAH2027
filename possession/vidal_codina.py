"""Rule-based player and team possession from tracking (Vidal-Codina et al., 2022).

Port of databallpy 0.8.1 `TrackingData.add_individual_player_possession`
(MIT licence, https://github.com/alek050/databallpy), which implements
Vidal-Codina, Evans, El Fakir & Billingham, "Automatic event detection in football using
tracking data", Sports Engineering 25 (2022). It is ported rather than imported because
databallpy pins numpy < 2.3 and pandas < 3, which would change our pinned environment.

The rule:
1. Possession zone: the player closest to the ball, if within `pz_radius` m, is a candidate.
2. A candidate run becomes a *gain* if, while the ball is in that player's zone, the ball's
   speed changes by more than `bv_threshold` m/s between consecutive frames, or its direction
   changes by more than `ba_threshold` degrees between entering and leaving the zone.
3. The possession lasts until the ball leaves the zone. If the same player is the next
   candidate, the possession is extended (databallpy behaviour). One deliberate difference:
   databallpy writes that extension to the previous *run*, which drops it when that run was
   another player's non-gain; here it always extends the last valid gain.

`team_codes` turns player possessions into H/A/D/X per frame. A team holds the ball from its
gain until the other team gains, and that includes passes in flight and frames where the ball
is missing. Holding stops at predicted dead balls. The first gain in a live interval is
back-filled to the interval's start, and an interval with no gain stays X.
"""

from dataclasses import dataclass

import numpy as np

from dead_ball.pipeline import A, D, H, X


@dataclass(frozen=True)
class VCParams:
    pz_radius: float = 1.5       # m (databallpy default)
    bv_threshold: float = 5.0    # m/s change between consecutive frames
    ba_threshold: float = 10.0   # degrees
    min_frames_pz: int = 0
    # 1 = the paper: speed jump between consecutive frames, direction over single-frame steps.
    # k > 1 = smoothing-robust variant (ours): speed change (max - min) over the zone run +- k
    # frames, direction from the k-frame step before entering to the k-frame step after leaving.
    window_frames: int = 1


# Chosen for our ATD: IDSSE with the ball smoothed like ATD's (sigma 0.25 s) and BAR-ATH-calibrated
# blackouts, leave-one-match-out (analysis/09 --grid refine; 2.0 m in 7/7 folds;
# observations/2026-11-05_blackout-calibration.md). On a smoothed ball the speed trigger never
# fires, so gains come from the direction trigger alone.
TUNED = VCParams(pz_radius=2.0, bv_threshold=5.0, ba_threshold=10.0, window_frames=1)


def ball_speed(ball, fps):
    """Backward-difference speed in m/s; NaN where either frame is missing."""
    s = np.full(len(ball), np.nan)
    s[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    return s


def _runs_of(values):
    """(start, end_inclusive, value) for each run of equal values."""
    if not len(values):
        return []
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges - 1, len(values) - 1]
    return [(s, e, values[s]) for s, e in zip(starts, ends)]


def _angle_deg(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if not (np.isfinite(nu) and np.isfinite(nv)) or nu == 0 or nv == 0:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)))


def player_possession(ball, players, fps, params=VCParams(), gains=None):
    """Per-frame index into `players` (n, P, 2) of the player in possession, -1 for nobody.
    If `gains` is a list, (start_frame, speed_ok, angle_ok) is appended for each valid gain."""
    n = len(ball)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    closest = dist.argmin(axis=1)
    candidate = np.where(dist.min(axis=1) < params.pz_radius, closest, -1)

    speed = ball_speed(ball, fps)
    jump = np.zeros(n, bool)
    jump[1:] = np.abs(np.diff(speed)) > params.bv_threshold          # NaN compares False

    owner = np.full(n, -1)
    last_player, last_end = -1, -1
    for s, e, p in _runs_of(candidate):
        if p < 0:
            continue
        if p == last_player:                       # same player again: extend
            owner[last_end + 1:e + 1] = p
            last_end = e
            continue
        k = params.window_frames
        if k <= 1:
            speed_ok = jump[s:e + 1].any()
        else:
            w = speed[max(s - k, 0):min(e + k + 1, n)]
            w = w[np.isfinite(w)]
            speed_ok = bool(len(w) and w.max() - w.min() > params.bv_threshold)
        k = max(k, 1)
        angle = _angle_deg(ball[s] - ball[max(s - k, 0)], ball[min(e + k, n - 1)] - ball[e])
        angle_ok = bool(np.isfinite(angle) and angle > params.ba_threshold)
        if e - s + 1 >= params.min_frames_pz and (speed_ok or angle_ok):
            owner[s:e + 1] = p
            last_player, last_end = p, e
            if gains is not None:
                gains.append((int(s), bool(speed_ok), angle_ok))
    return owner


def team_codes(owner, player_team, dead):
    """H/A/D/X per frame from player possession (`player_team`: 0 home, 1 away per player)."""
    codes = np.full(len(owner), X, np.int8)
    codes[dead] = D
    team = np.where(owner >= 0, np.asarray(player_team)[np.maximum(owner, 0)], -1)
    for s, e, is_dead in _runs_of(np.asarray(dead, bool)):
        if is_dead:
            continue
        seg = team[s:e + 1]
        gained = np.flatnonzero(seg >= 0)
        if not len(gained):
            continue
        idx = np.maximum.accumulate(np.where(seg >= 0, np.arange(len(seg)), -1))
        idx[:gained[0]] = gained[0]                   # back-fill to the interval start
        codes[s:e + 1] = np.where(seg[idx] == 0, H, A)
    return codes


def period_codes(p, dead, fps, params=VCParams()):
    """Convenience: a dead_ball.idsse.Period (centre-origin metres) → H/A/D/X codes."""
    players = np.concatenate([p.home, p.away], axis=1)
    team = np.r_[np.zeros(p.home.shape[1], int), np.ones(p.away.shape[1], int)]
    return team_codes(player_possession(p.ball, players, fps, params), team, dead)
