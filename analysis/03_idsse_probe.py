"""Can IDSSE (DFL, other games) supply dead-ball labels? Aggregates only.

    uv run python analysis/03_idsse_probe.py [MATCH_ID]

Files are fetched from the HuggingFace mirror (pysport/idsse-data) into data/external/idsse/
(gitignored). The figshare URLs baked into kloppy return an empty body to scripted clients.

Checks, per match:
  - frame rate, periods, frames per period
  - BallStatus (alive/dead) and BallPossession (home/away) as kloppy exposes them
  - dead share, dead-spell count and durations, short flickers
  - possession share and changes while alive
  - ball z availability, players per frame
"""

import os
import sys

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "external", "idsse")
COMP = {"J03WMX": "000001", "J03WN1": "000001"}   # all other matches are 2. Bundesliga
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))   # inclusive (start, end)


def q(values):
    return " / ".join(f"{v:.1f}" for v in np.quantile(values, QUANTILES)) if len(values) else "-"


def main(match_id):
    tag = f"DFL-COM-{COMP.get(match_id, '000002')}_DFL-MAT-{match_id}.xml"
    ds = sportec.load_tracking(
        raw_data=os.path.join(DIR, f"DFL_04_03_positions_raw_observed_{tag}"),
        meta_data=os.path.join(DIR, f"DFL_02_01_matchinformation_{tag}"),
        only_alive=False,   # default True silently drops every dead-ball frame
    )
    fps = ds.metadata.frame_rate
    home = ds.metadata.teams[0]
    print(f"match {match_id}: fps {fps}, frames {len(ds.frames)}, periods {len(ds.metadata.periods)}")

    for period in ds.metadata.periods:
        frames = [f for f in ds.frames if f.period.id == period.id]
        n = len(frames)
        dead = np.array([f.ball_state == BallState.DEAD for f in frames])
        owner_h = np.array([f.ball_owning_team == home for f in frames])
        ball_ok = np.array([f.ball_coordinates is not None for f in frames])
        ball_z = np.array([f.ball_coordinates is not None and f.ball_coordinates.z not in (None, 0)
                           for f in frames])
        n_players = np.array([len(f.players_data) for f in frames])

        spells = [(e - s + 1) / fps for s, e in runs(dead)]
        alive_spells = [(e - s + 1) / fps for s, e in runs(~dead)]
        alive = ~dead
        flips = int(np.sum(owner_h[1:][alive[1:] & alive[:-1]] != owner_h[:-1][alive[1:] & alive[:-1]]))
        print(f"\n period {period.id}: {n} frames ({n / fps / 60:.1f} min)")
        print(f"  dead share {dead.mean():.1%}; dead spells {len(spells)}, "
              f"duration q10/25/50/75/90 s = {q(spells)}; spells < 1 s: {sum(s < 1 for s in spells)}")
        print(f"  alive spells {len(alive_spells)}, duration q10/25/50/75/90 s = {q(alive_spells)}")
        print(f"  while alive: home owns {owner_h[alive].mean():.1%}, owner changes {flips}")
        print(f"  ball position present {ball_ok.mean():.1%}, ball z non-zero {ball_z.mean():.1%}")
        print(f"  players per frame q10/50/90 = {q(n_players)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "J03WPY")
