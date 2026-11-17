"""IDSSE (DFL) matches as numpy arrays: labels, ball, per-team player slots. Cached in artifacts/.

Nothing here prints per-frame data. Other games only (IDSSE / DFL, CC BY 4.0).
"""

import glob
import importlib.util
import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE = os.path.join(ROOT, "data", "external", "idsse")
CACHE = os.path.join(ROOT, "artifacts")

MATCH_IDS = sorted({os.path.basename(p).split("DFL-MAT-")[1][:-4]
                    for p in glob.glob(os.path.join(IDSSE, "DFL_04_03_positions_raw_observed_*.xml"))})


def _conv():
    spec = importlib.util.spec_from_file_location(
        "conv", os.path.join(ROOT, "analysis", "04_dead_ball_conventions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Period:
    t: np.ndarray          # (n,) seconds
    dead: np.ndarray       # (n,) bool, DFL flag
    home_owns: np.ndarray  # (n,) bool, DFL BallPossession == home
    ball: np.ndarray       # (n, 2) float32 metres, centre origin; NaN = missing
    home: np.ndarray       # (n, Ph, 2) float32; NaN = slot not on the pitch
    away: np.ndarray       # (n, Pa, 2) float32
    home_gk: np.ndarray    # (n,) int slot index of the goalkeeper on the pitch, -1 if none
    away_gk: np.ndarray

    def __len__(self):
        return len(self.t)


@dataclass
class Match:
    id: str
    fps: float
    half_len: float
    half_wid: float
    periods: list


def _gk_index(xy, gk_slots):
    out = np.full(len(xy), -1, dtype=np.int16)
    for s in gk_slots:
        present = np.isfinite(xy[:, s, 0])
        out[present & (out < 0)] = s
    return out


def _build(match_id):
    from kloppy import sportec
    from kloppy.domain import BallState, PositionType

    pos, meta, _ = _conv().dfl_files(match_id)
    ds = sportec.load_tracking(raw_data=pos, meta_data=meta, only_alive=False, coordinates="sportec")
    home_team, away_team = ds.metadata.teams
    slots = {home_team: {}, away_team: {}}
    gks = {home_team: [], away_team: []}
    for team in (home_team, away_team):
        for i, pl in enumerate(team.players):
            slots[team][pl.player_id] = i
            if pl.starting_position == PositionType.Goalkeeper:
                gks[team].append(i)
    dims = ds.metadata.pitch_dimensions
    arrays = dict(fps=ds.metadata.frame_rate, half_len=dims.x_dim.max, half_wid=dims.y_dim.max)
    for period in ds.metadata.periods:
        fr = [f for f in ds.frames if f.period.id == period.id]
        n = len(fr)
        p = period.id
        ball = np.full((n, 2), np.nan, np.float32)
        xy = {tm: np.full((n, len(slots[tm]), 2), np.nan, np.float32) for tm in slots}
        for k, f in enumerate(fr):
            if f.ball_coordinates is not None:
                ball[k] = (f.ball_coordinates.x, f.ball_coordinates.y)
            for pl, d in f.players_data.items():
                if d.coordinates is not None:
                    xy[pl.team][k, slots[pl.team][pl.player_id]] = (d.coordinates.x, d.coordinates.y)
        arrays[f"t{p}"] = np.array([f.timestamp.total_seconds() for f in fr])
        arrays[f"dead{p}"] = np.array([f.ball_state == BallState.DEAD for f in fr])
        arrays[f"home_owns{p}"] = np.array([f.ball_owning_team == home_team for f in fr])
        arrays[f"ball{p}"] = ball
        arrays[f"home{p}"] = xy[home_team]
        arrays[f"away{p}"] = xy[away_team]
        arrays[f"home_gk{p}"] = _gk_index(xy[home_team], gks[home_team])
        arrays[f"away_gk{p}"] = _gk_index(xy[away_team], gks[away_team])
    return arrays


def load_match(match_id: str) -> Match:
    path = os.path.join(CACHE, f"idsse_match_{match_id}.npz")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, **_build(match_id))
    z = np.load(path)
    ids = sorted(int(k[1:]) for k in z.files if k.startswith("t") and k[1:].isdigit())
    fields = ("t", "dead", "home_owns", "ball", "home", "away", "home_gk", "away_gk")
    periods = [Period(**{f: z[f"{f}{p}"] for f in fields}) for p in ids]
    return Match(match_id, float(z["fps"]), float(z["half_len"]), float(z["half_wid"]), periods)
