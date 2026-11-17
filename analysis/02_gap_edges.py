"""What happens around the ball blackouts? Aggregates only; per-gap table -> artifacts/gap_edges.csv.

    uv run python analysis/02_gap_edges.py

For every gap (a maximal run of graded frames with no ball position, inside one half):
  - nearest team to the ball just before the gap and just after it (majority over EDGE_FRAMES
    visible frames), and how clear that was (distance margin, metres)
  - is the ball over / on a line when it disappears or reappears
  - is the ball stationary right after it reappears (placed for a restart)
  - do all players slow down somewhere inside the gap (stoppage signature)

No ground truth is used. Thresholds are generic football constants, not tuned to this match.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.atd_cache import load  # noqa: E402

# Params
FPS = 25
EDGE_FRAMES = 25            # visible frames used on each side of a gap (1 s)
LINE_MARGIN_M = 1.0         # ball within this of a touch/goal line counts as "on the line"
STATIC_DISPLACEMENT_M = 1.0  # ball moves less than this over its first EDGE_FRAMES visible frames
SPEED_WINDOW = 2 * FPS      # rolling window for the team-wide mean speed
STOPPAGE_SPEED = 1.0        # m/s: rolling mean speed of all tracked players below this = stoppage cue
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
OUT = os.path.join(ROOT, "artifacts", "gap_edges.csv")


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))   # inclusive (start, end)


def nearest_team(atd, frames):
    """Majority nearest team (0 = H, 1 = A) and median margin in metres over the given frames."""
    if len(frames) == 0:
        return -1, np.nan
    dx = atd["px"][frames] - atd["ball_x"][frames, None]
    dy = atd["py"][frames] - atd["ball_y"][frames, None]
    d = np.hypot(dx, dy)
    team = atd["slot_team"]
    d_h = np.nanmin(np.where(team == 0, d, np.nan), axis=1)
    d_a = np.nanmin(np.where(team == 1, d, np.nan), axis=1)
    votes = np.where(d_h <= d_a, 0, 1)
    winner = int(np.round(votes.mean())) if votes.mean() != 0.5 else 0
    return winner, float(np.nanmedian(np.abs(d_h - d_a)))


def on_line(x, y, length, width):
    return bool(x < LINE_MARGIN_M or x > length - LINE_MARGIN_M
                or y < LINE_MARGIN_M or y > width - LINE_MARGIN_M)


def q(values, unit=""):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n=0"
    return "n={:<4} ".format(values.size) + " ".join(
        "q{:g}={:.1f}{}".format(k, np.quantile(values, k), unit) for k in QUANTILES)


if __name__ == "__main__":
    atd = load()
    length, width = atd["pitch"]
    n = len(atd["ball_x"])
    visible = np.isfinite(atd["ball_x"])

    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(atd["ps"], axis=1)
    rolling_speed = pd.Series(mean_speed).rolling(SPEED_WINDOW, center=True, min_periods=FPS).mean().values

    rows = []
    for half, first, last in atd["halves"]:
        in_half = np.zeros(n, bool)
        in_half[first:last + 1] = True
        for start, end in runs(in_half & ~visible):
            before = np.flatnonzero(visible[first:start])[-EDGE_FRAMES:] + first
            after = np.flatnonzero(visible[end + 1:last + 1])[:EDGE_FRAMES] + end + 1
            team_before, margin_before = nearest_team(atd, before)
            team_after, margin_after = nearest_team(atd, after)
            row = {
                "half": half, "start": start, "end": end, "dur_s": (end - start + 1) / FPS,
                "at_half_edge": start == first or end == last,
                "team_before": team_before, "team_after": team_after,
                "margin_before_m": margin_before, "margin_after_m": margin_after,
                "exit_on_line": on_line(atd["ball_x"][before[-1]], atd["ball_y"][before[-1]], length, width)
                if len(before) else np.nan,
                "reentry_on_line": on_line(atd["ball_x"][after[0]], atd["ball_y"][after[0]], length, width)
                if len(after) else np.nan,
                "reentry_static": bool(np.hypot(atd["ball_x"][after[-1]] - atd["ball_x"][after[0]],
                                                atd["ball_y"][after[-1]] - atd["ball_y"][after[0]])
                                       < STATIC_DISPLACEMENT_M) if len(after) >= 2 else np.nan,
                "min_rolling_speed": float(np.nanmin(rolling_speed[start:end + 1])),
            }
            row["stoppage_cue"] = row["min_rolling_speed"] < STOPPAGE_SPEED
            rows.append(row)

    gaps = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gaps.to_csv(OUT, index=False)

    # ---------------------------------------------------------------- aggregates
    graded = np.zeros(n, bool)
    for _, first, last in atd["halves"]:
        graded[first:last + 1] = True
    print("gaps: {}   missing graded frames: {} ({:.1%} of graded)".format(
        len(gaps), int((gaps.dur_s * FPS).sum()), (gaps.dur_s * FPS).sum() / graded.sum()))
    print("gap duration:", q(gaps.dur_s, "s"))
    print("gaps touching a half edge (no 'before' or 'after' side): {}".format(int(gaps.at_half_edge.sum())))

    print("\nteam-wide rolling mean speed (2 s), m/s, graded frames")
    print("  ball visible:", q(rolling_speed[graded & visible]))
    print("  ball missing:", q(rolling_speed[graded & ~visible]))
    print("  min inside each gap:", q(gaps.min_rolling_speed))
    print("  stoppage cue threshold: {} m/s".format(STOPPAGE_SPEED))

    inner = gaps[~gaps.at_half_edge].copy()
    inner["team_change"] = inner.team_before != inner.team_after
    print("\nedge clarity: nearest-team margin, metres (small = contested at the edge)")
    print("  before:", q(inner.margin_before_m))
    print("  after: ", q(inner.margin_after_m))

    print("\ncue rates over {} interior gaps (share of gaps | share of their missing frames)".format(len(inner)))
    w = inner.dur_s
    for col in ["team_change", "exit_on_line", "reentry_on_line", "reentry_static", "stoppage_cue"]:
        m = inner[col].astype(bool)
        print("  {:<16} {:5.1%} | {:5.1%}".format(col, m.mean(), w[m].sum() / w.sum()))

    inner["restart_cue"] = (inner.exit_on_line.astype(bool) | inner.reentry_on_line.astype(bool)
                            | inner.reentry_static.astype(bool) | inner.stoppage_cue)
    print("\ncategories (restart_cue = any of: exit/reentry on line, static reentry, stoppage)")
    print("  {:<34} {:>5} {:>8}  duration".format("", "gaps", "frames%"))
    for change in (False, True):
        for cue in (False, True):
            m = (inner.team_change == change) & (inner.restart_cue == cue)
            label = "{} team, {}".format("different" if change else "same",
                                          "restart cue" if cue else "no restart cue")
            print("  {:<34} {:>5} {:>7.1%}  {}".format(label, int(m.sum()), w[m].sum() / w.sum(), q(w[m], "s")))
    print("\nper-gap table (local only): {}".format(os.path.relpath(OUT, ROOT)))
