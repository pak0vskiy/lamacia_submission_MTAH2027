"""Kickoff detection from player positions (Laws of the Game, Law 8), no ball needed.

At a kickoff every player is in his own half. In live play that almost never holds for long,
because attackers of at least one team are in the other half. So:

- separated(frame): both teams have >= MIN_PLAYERS on the pitch, and one team is entirely at
  x <= +tol while the other is entirely at x >= -tol (players stand on the line), up to
  MAX_OVER outliers;
- formation = separated at the loose FORM_TOL_M, with breaks up to CLOSE_S closed;
- the kickoff is the first formation run of >= MIN_FORM_S after which separation is (almost) gone
  for FOLLOW_S, i.e. it broke because play started. The kick is placed one frame after the last
  frame of that run that is separated at the strict TOL_M (the loose tolerance ends late),
  minus LAG_S (the formation breaks a little after the kick).

Generic: no frame, timestamp or match constant. Tuned and checked on other games only
(analysis/17_kickoff_detector.py).
"""

import numpy as np

TOL_M = 1.0          # strict: players allowed this far over the halfway line (timing)
FORM_TOL_M = 2.0     # loose: for finding the formation
CLOSE_S = 1.0        # breaks in the formation up to this long are closed
MIN_PLAYERS = 8      # per team, on the pitch
MAX_OVER = 1         # players allowed over the line (outliers)
MIN_FORM_S = 3.0     # a kickoff formation lasts at least this long
FOLLOW_S = 20.0      # after the kick, separation is rare for this long
FOLLOW_MAX = 0.1     # ... at most this share of frames
LAG_S = 0.6          # detected break - true kick, pooled median on IDSSE + Metrica (analysis/17)


def separated(home, away, tol=TOL_M, min_players=MIN_PLAYERS, max_over=MAX_OVER):
    """(n,) bool: the two teams stand in opposite halves, allowing `max_over` players (both teams
    together) more than `tol` over the halfway line (tracking outliers, mis-assigned tracks)."""
    hx, ax = home[..., 0], away[..., 0]
    hn, an = np.isfinite(hx).sum(1), np.isfinite(ax).sum(1)
    enough = (hn >= min_players) & (an >= min_players)
    home_left = (hx > tol).sum(1) + (ax < -tol).sum(1)     # NaN compares False
    home_right = (hx < -tol).sum(1) + (ax > tol).sum(1)
    return enough & (np.minimum(home_left, home_right) <= max_over)


def detect(home, away, fps, start=0, stop=None, lag_s=LAG_S):
    """Frame index of the first kickoff in [start, stop), or None.

    Returns the first frame after the formation breaks (the kick), i.e. frames [start, k) are the
    pre-kickoff stoppage."""
    stop = len(home) if stop is None else stop
    h, a = home[start:stop], away[start:stop]
    loose, strict = separated(h, a, FORM_TOL_M), separated(h, a, TOL_M)
    form = _close(loose, int(round(CLOSE_S * fps)))
    follow = int(round(FOLLOW_S * fps))
    for s, e in _runs(form):
        if e - s < MIN_FORM_S * fps:
            continue
        after = loose[e:e + follow]
        if len(after) and after.mean() <= FOLLOW_MAX:
            hit = np.flatnonzero(strict[s:e])
            k = s + hit[-1] + 1 if len(hit) else e
            return start + max(0, k - int(round(lag_s * fps)))
    return None


def _runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))   # [start, end)


def _close(mask, gap):
    """Fill False runs of <= gap frames that sit between True runs."""
    out = mask.copy()
    runs = _runs(mask)
    for (_, e0), (s1, _) in zip(runs[:-1], runs[1:]):
        if s1 - e0 <= gap:
            out[e0:s1] = True
    return out
