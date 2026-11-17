"""Aggregate statistics of the tactical ATD. Prints quantiles and rates only.

    uv run python analysis/01_atd_aggregates.py

Settles: speed unit (m/s vs km/h), how often the ball speed channel is present when the position
is, how often the ball sits outside the 0-1 pitch box, tracked objects per team, and the metadata
parameter *names* (not values).
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths, half_video_blocks  # noqa: E402

# Params
QUANTILES = [0.5, 0.9, 0.99, 0.999]


def strip_ns(tag):
    return tag.split("}", 1)[-1]


def meta(path):
    root = ET.parse(path).getroot()
    names = collections.Counter()
    slot_team = []
    for node in root.iter():
        tag = strip_ns(node.tag)
        if tag == "ProviderParameter":
            names[node.findtext("{*}Name") or node.findtext("Name")] += 1
        if tag == "Player":
            slot_team.append(node.attrib.get("teamId"))
    teams = [strip_ns(t.tag) and t.attrib.get("id") for t in root.iter() if strip_ns(t.tag) == "Team"]
    print("provider parameter names (count):", dict(names))
    print("team ids in order: {} -> first is home (H)".format(["team#{}".format(i) for i, _ in enumerate(teams)]))
    per_team = collections.Counter(teams.index(t) if t in teams else -1 for t in slot_team)
    print("track slots per team index:", dict(per_team))
    return [teams.index(t) if t in teams else -1 for t in slot_team]


def load_raw(path, n_slots):
    n = sum(1 for _ in open(path))
    px = np.full((n, n_slots), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    ball = np.full((n, 3), np.nan, np.float32)
    with open(path) as handle:
        for i, line in enumerate(handle):
            _, players, b = line.rstrip("\n").split(":")
            for j, g in enumerate(players.split(";")[:n_slots]):
                x, y, s = g.split(",")
                px[i, j], py[i, j], ps[i, j] = float(x), float(y), float(s)
            ball[i] = [float(v) for v in b.strip().rstrip(";").split(",")]
    return px, py, ps, ball


def q(arr):
    arr = arr[np.isfinite(arr)]
    return "n={:>9}  ".format(arr.size) + "  ".join(
        "q{:g}={:.3f}".format(k, np.quantile(arr, k)) for k in QUANTILES)


if __name__ == "__main__":
    paths = asset_paths()
    slot_team = meta(paths["atd_meta"])
    px, py, ps, ball = load_raw(paths["atd_raw"], len(slot_team))

    graded = np.zeros(len(ball), bool)
    for _, a, b in half_video_blocks("BAR-ATH"):
        graded[a:b + 1] = True   # ATD row i == video frame i (frame_id - 1)

    print("\nplayer speed (graded frames):", q(ps[graded]))
    print("  -> if q0.99 is ~25-35 the unit is km/h; if ~7-10 it is m/s")
    print("player x range:", q(px[graded]))
    print("ball speed   (graded):", q(ball[graded, 2]))

    has_pos = np.isfinite(ball[:, 0])
    has_spd = np.isfinite(ball[:, 2])
    print("\nball position present: all {:.1%}  graded {:.1%}".format(has_pos.mean(), has_pos[graded].mean()))
    print("ball speed present | position present: {:.1%}".format(has_spd[has_pos].mean()))
    out = has_pos & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    print("ball outside 0-1 box | position present (graded): {:.2%}".format(out[graded & has_pos].mean()))

    # ball gap lengths on graded frames, in seconds
    g = (~has_pos) & graded
    edges = np.flatnonzero(np.diff(np.r_[0, g.astype(np.int8), 0]))
    lengths = (edges[1::2] - edges[::2]) / 25.0
    print("ball gaps (graded): count {}  ".format(lengths.size) + "  ".join(
        "q{:g}={:.1f}s".format(k, np.quantile(lengths, k)) for k in QUANTILES)
        + "  share of missing frames in gaps >5s: {:.1%}".format(lengths[lengths > 5].sum() / lengths.sum()))

    teams = np.array(slot_team)
    for t in (0, 1):
        cnt = np.isfinite(px[:, teams == t]).sum(1)
        print("tracked objects team#{} per graded frame: mean {:.2f}  q0.05 {:.0f}  q0.95 {:.0f}".format(
            t, cnt[graded].mean(), np.quantile(cnt[graded], 0.05), np.quantile(cnt[graded], 0.95)))
    unk = (teams < 0).sum()
    print("slots without a known team:", int(unk))
