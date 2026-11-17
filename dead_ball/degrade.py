"""Make clean IDSSE tracking look like our ATD: ball blackouts in long blocks, player drop-outs.

Gap durations follow BAR-ATH's aggregate gap distribution (observations/2026-11-02_ball-gaps.md).
How strongly blackouts coincide with dead ball is unknown on our match (it would need GT), so it is
a parameter: `coincidence` = probability that a gap is centred on a dead frame.
"""

from dataclasses import replace

import numpy as np

GAP_QUANTILES = (0.5, 6.1, 10.7, 21.1, 33.8)    # s, at GAP_PROBS
GAP_PROBS = (0.1, 0.25, 0.5, 0.75, 0.9)
GAP_CLAMP = (0.2, 60.0)
MISSING_SHARE = 0.41
BALL_JITTER_M = 0.02   # iid tracking noise on every ball position


def unfreeze_ball(ball, dead, rng, jitter_m=BALL_JITTER_M):
    """DFL freezes the ball position while dead (100 % of dead frames, 0.8 % of alive; our ATD 0.1 %).
    Replace each frozen spell by a straight carry from the freeze point to the restart point, then
    add tracking jitter everywhere, so ball stillness cannot stand in for the label."""
    out = ball.astype(np.float64).copy()
    n = len(out)
    edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        a = out[s - 1] if s > 0 else out[s]
        b = out[e] if e < n else out[e - 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        w = (np.arange(s, e) - s + 1) / (e - s + 1)
        out[s:e] = a + w[:, None] * (b - a)
    out += rng.normal(0, jitter_m, out.shape)
    return out.astype(np.float32)


STILL_JITTER_M = 0.005   # low tracking noise for the "mixed" motion model


def _leg(p0, p1, speed, fps):
    n = max(1, int(np.ceil(np.hypot(*(p1 - p0)) / speed * fps)))
    w = (np.arange(1, n + 1) / n)[:, None]
    return p0 + w * (p1 - p0)


def _spell_path(a, va, b, n, fps, rng):
    """n positions from the exit point a (velocity va) to the restart point b: overshoot past the
    line, then legs that are carries (0.8-2 m/s) or throws / kicks (6-15 m/s) through 0-2 random
    waypoints, with the remaining time spent still at random pauses."""
    speed = float(np.hypot(*va))
    direction = va / speed if speed > 0.5 else rng.normal(size=2) / np.sqrt(2)
    v0 = float(np.clip(speed, 2.0, 15.0))
    t_over = min(rng.uniform(0.5, 2.5), n / fps / 3)
    k = max(1, int(t_over * fps))
    t = (np.arange(1, k + 1) / fps)[:, None]
    over = a + direction * (v0 * t - v0 * t ** 2 / (2 * t_over))
    pts = [over[-1]] + [over[-1] + rng.normal(0, 8.0, 2) for _ in range(rng.integers(0, 3))] + [b]
    legs = []
    for p0, p1 in zip(pts[:-1], pts[1:]):
        fast = rng.random() < 0.4
        legs.append((p0, p1, rng.uniform(6, 15) if fast else rng.uniform(0.8, 2.0)))
    remaining = n - k
    moving = [_leg(p0, p1, v, fps) for p0, p1, v in legs]
    total = sum(len(m) for m in moving)
    if total > remaining:   # not enough time: speed every leg up
        scale = total / max(remaining, 1)
        moving = [_leg(p0, p1, v * scale * 1.05, fps) for p0, p1, v in legs]
        total = sum(len(m) for m in moving)
    pauses = np.floor(rng.dirichlet(np.ones(len(moving) + 1)) * max(remaining - total, 0)).astype(int)
    out, cur = [over], over[-1]
    for pause, m in zip(pauses, moving + [None]):
        out.append(np.repeat(cur[None], pause, axis=0))
        if m is not None:
            out.append(m)
            cur = m[-1]
    path = np.concatenate(out)[:n]
    if len(path) < n:
        path = np.concatenate([path, np.repeat(path[-1:], n - len(path), axis=0)])
    return path


def dead_ball_path(ball, dead, rng, fps=25.0, jitter_m=STILL_JITTER_M):
    """'mixed' motion model: replaces DFL's frozen dead ball by overshoot + carries / throws /
    pauses ending at the restart point, plus low tracking jitter everywhere."""
    out = ball.astype(np.float64).copy()
    n = len(out)
    edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        a = out[s - 1] if s > 0 else out[s]
        b = out[e] if e < n else out[e - 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()) or e - s < 3:
            continue
        lag = min(5, s - 1) if s > 1 else 0
        va = (out[s - 1] - out[s - 1 - lag]) * fps / lag if lag else np.zeros(2)
        va = va if np.isfinite(va).all() else np.zeros(2)
        out[s:e] = _spell_path(a, va, b, e - s, fps, rng)
    out += rng.normal(0, jitter_m, out.shape)
    return out.astype(np.float32)


def sample_gap_s(rng, n):
    """Piecewise-linear inverse CDF through the quantiles; tails extended linearly, clamped."""
    p = np.r_[0.0, GAP_PROBS, 1.0]
    lo = GAP_QUANTILES[0] - (GAP_QUANTILES[1] - GAP_QUANTILES[0]) / (GAP_PROBS[1] - GAP_PROBS[0]) * GAP_PROBS[0]
    hi = GAP_QUANTILES[-1] + (GAP_QUANTILES[-1] - GAP_QUANTILES[-2]) / (GAP_PROBS[-1] - GAP_PROBS[-2]) * (1 - GAP_PROBS[-1])
    x = np.r_[lo, GAP_QUANTILES, hi]
    return np.clip(np.interp(rng.random(n), p, x), *GAP_CLAMP)


def ball_mask(dead, fps, rng, coincidence, missing_share=MISSING_SHARE):
    """Bool mask, True = ball hidden. Gaps placed until `missing_share` of frames are hidden."""
    n = len(dead)
    mask = np.zeros(n, bool)
    target = int(round(missing_share * n))
    dead_idx = np.flatnonzero(dead)
    hidden = 0
    while hidden < target:
        length = max(1, int(round(sample_gap_s(rng, 1)[0] * fps)))
        if len(dead_idx) and rng.random() < coincidence:
            centre = dead_idx[rng.integers(len(dead_idx))]
        else:
            centre = rng.integers(n)
        s = max(0, centre - length // 2)
        e = min(n, s + length)
        new = int((~mask[s:e]).sum())
        if hidden + new > target:   # trim the last gap so the share lands on target
            free = np.flatnonzero(~mask[s:e])
            e = s + free[target - hidden - 1] + 1
            new = target - hidden
        mask[s:e] = True
        hidden += new
    return mask


# ---- 'calibrated' blackouts: fitted to BAR-ATH tracking, label-free -----------------------------
# analysis/12_blackout_calibration.py, with live / dead = our dead-ball model's calls on BAR-ATH.
# Quantiles at CAL_PROBS. See observations/2026-11-05_blackout-calibration.md.
CAL_PROBS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
CAL_MIN_SPELL_S = 2.0
CAL_P_COVER = 0.95                                          # dead spells with a covering gap
CAL_START_OFF = (-6.34, -5.52, -2.53, 0.06, 3.01, 12.90, 17.96)   # gap start - spell start, s
CAL_END_OFF = (-13.01, -11.08, -5.24, -0.82, 0.06, 1.98, 3.79)    # gap end - spell end, s
CAL_LIVE_RATE = 0.64                                         # live-only gaps per live minute
CAL_LIVE_LEN = (0.07, 0.08, 0.12, 5.36, 7.04, 9.49, 11.27)   # live-only gap length, s
CAL_DUEL_M = 3.0
CAL_DUEL_BETA = 1.5          # start weight 1 + beta * duel; fitted on IDSSE so the duel ratio is ~1.9


def _inv_cdf(rng, table, n=None):
    """Sample through a quantile table (piecewise linear, clamped at the outer quantiles)."""
    return np.interp(rng.random(n), CAL_PROBS, table)


def duel_frames(p, radius=CAL_DUEL_M):
    """True where >= 1 player of each team is within `radius` m of the (clean) ball."""
    def near(xy):
        d = np.linalg.norm(xy - p.ball[:, None, :], axis=2)
        return (np.nan_to_num(d, nan=np.inf) < radius).any(axis=1)
    return near(p.home) & near(p.away)


def calibrated_mask(p, fps, rng, duel_beta=CAL_DUEL_BETA):
    """Bool mask, True = ball hidden. Gaps anchored to dead spells, plus short live-only gaps
    placed preferentially at duels, with rates and lengths measured on BAR-ATH."""
    dead = p.dead
    n = len(dead)
    mask = np.zeros(n, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < CAL_MIN_SPELL_S * fps or rng.random() >= CAL_P_COVER:
            continue
        g0 = int(round(a + _inv_cdf(rng, CAL_START_OFF) * fps))
        g1 = int(round(b + _inv_cdf(rng, CAL_END_OFF) * fps))
        g0, g1 = max(g0, 0), min(g1, n)
        if g1 > g0:
            mask[g0:g1] = True
    live = ~dead
    n_live = int(rng.poisson(CAL_LIVE_RATE * live.sum() / fps / 60))
    cand = np.flatnonzero(live & ~mask)
    if n_live and len(cand):
        w = 1.0 + duel_beta * duel_frames(p)[cand]
        starts = rng.choice(cand, size=n_live, p=w / w.sum())
        lengths = np.maximum(1, np.round(_inv_cdf(rng, CAL_LIVE_LEN, n_live) * fps)).astype(int)
        for s, k in zip(starts, lengths):
            mask[s:min(n, s + k)] = True
    return mask


def _drop_players(xy, fps, rng, rate):
    out = xy.copy()
    n = len(xy)
    for slot in range(xy.shape[1]):
        present = np.isfinite(xy[:, slot, 0])
        goal = rate * present.sum()
        removed = 0
        while removed < goal:
            length = int(rng.uniform(1, 5) * fps)
            s = rng.integers(n)
            e = min(n, s + length)
            removed += int(np.isfinite(out[s:e, slot, 0]).sum())
            out[s:e, slot] = np.nan
    return out


def degrade(p, fps, rng, coincidence, player_dropout=0.02, motion="carry", blackout="coincidence"):
    """Copy of Period with ball blackouts and player drop-outs. Labels untouched.
    motion: how the (DFL-frozen) dead ball moves - 'carry' (straight carry + 2 cm jitter) or
    'mixed' (overshoot, carries, throws, pauses + 0.5 cm jitter).
    blackout: 'coincidence' (41 % of frames, a share `coincidence` of gaps centred on dead frames)
    or 'calibrated' (live / dead rates and gap lengths fitted to BAR-ATH; `coincidence` ignored)."""
    if blackout == "calibrated":
        mask = calibrated_mask(p, fps, rng)
    else:
        mask = ball_mask(p.dead, fps, rng, coincidence)
    ball = (dead_ball_path(p.ball, p.dead, rng, fps) if motion == "mixed"
            else unfreeze_ball(p.ball, p.dead, rng))
    ball[mask] = np.nan
    return replace(p, ball=ball,
                   home=_drop_players(p.home, fps, rng, player_dropout),
                   away=_drop_players(p.away, fps, rng, player_dropout))


def smooth_ball(ball, fps, sigma_s):
    """Gaussian-smooth the ball track within each visible segment (gaps stay NaN).

    Our ATD ball changes speed far more smoothly than DFL's (live-play frame-to-frame speed
    change q99 1.2 vs 12.4 m/s; observations/2026-11-05_vidal-codina-rules.md), which looks like
    provider smoothing. sigma_s = 0 returns the input unchanged."""
    if sigma_s <= 0:
        return ball
    from scipy.ndimage import gaussian_filter1d
    out = np.array(ball, dtype=np.float32, copy=True)
    vis = np.isfinite(out).all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[0, vis.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        out[s:e] = gaussian_filter1d(out[s:e].astype(np.float64), sigma_s * fps, axis=0, mode="nearest")
    return out
