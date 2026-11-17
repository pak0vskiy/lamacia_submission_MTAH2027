"""Metrica Sports sample games 1-3 as numpy arrays, in the same shape as dead_ball.idsse.Match.

Other games only (public sample data, github.com/metrica-sports/sample-data). The open data has
no dead-ball flag and no owner, so both labels come from Metrica's own event data:

- dead: the event recipe of analysis/04 (D = [stoppage start, set piece)), plus everything before
  the first and after the last event of each period;
- owner: the team of the last on-ball event (PASS, SHOT, SET PIECE, RECOVERY, BALL LOST),
  forward-filled. CHALLENGE, BALL OUT, FAULT RECEIVED and CARD don't set the owner, so contested
  and loose phases stay with the last team, as CHALLENGE.md asks.

Coordinates: metres, centre origin, x along the pitch, y up (pitch 105 x 68 unless the metadata
says otherwise). Cached in artifacts/. Nothing here prints per-frame data.
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd

from dead_ball.idsse import Match, Period

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICA = os.path.join(ROOT, "data", "external", "metrica")
CACHE = os.path.join(ROOT, "artifacts")
GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)
OWNER_TYPES = {"PASS", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST"}


def _conv():
    spec = importlib.util.spec_from_file_location(
        "conv", os.path.join(ROOT, "analysis", "04_dead_ball_conventions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def events(g):
    """period, frame, team ('home' / 'away' / None), type, sub; sorted."""
    if g in (1, 2):
        df = pd.read_csv(os.path.join(METRICA, f"Sample_Game_{g}_RawEventsData.csv"))
        ev = pd.DataFrame(dict(period=df["Period"], frame=df["Start Frame"], frame_end=df["End Frame"],
                               team=df["Team"].str.lower(), type=df["Type"], sub=df["Subtype"].fillna("")))
    else:
        raw = json.load(open(os.path.join(METRICA, "Sample_Game_3_events.json")))["data"]

        def sub_name(s):
            if not s:
                return ""
            return " ".join(x["name"] for x in s) if isinstance(s, list) else s["name"]

        ev = pd.DataFrame([dict(period=e["period"], frame=e["start"]["frame"], frame_end=e["end"]["frame"],
                                team=(e.get("team") or {}).get("id"), type=e["type"]["name"],
                                sub=sub_name(e["subtypes"])) for e in raw])
    return ev.sort_values(["period", "frame"], kind="stable").reset_index(drop=True)


def _recipe_input(ev, fps):
    """The analysis/04 Metrica preprocessing, on frame-derived times."""
    ev = ev.copy()
    ev["t"] = ev.frame / fps
    ev["t_end"] = ev.frame_end / fps
    goal_kick = ev["sub"].str.startswith("GOAL KICK")
    ev["set_piece"] = np.where(ev.type == "SET PIECE", ev["sub"].str.replace(" ", "_"),
                               np.where(goal_kick, "GOAL_KICK", None))
    ev["set_piece"] = ev.set_piece.where(ev.set_piece.notna(), None)
    ev["play"] = ~ev.type.isin({"CARD", "BALL OUT"})
    ev["trigger"] = ((ev.type == "BALL OUT") & ~goal_kick) | (ev.type == "FAULT RECEIVED") \
        | ev["sub"].str.contains("OFFSIDE|GOAL$|-GOAL", regex=True)
    ev["t_stop"] = ev[["t", "t_end"]].max(axis=1)
    return ev


def labels(ev, frames, period, fps):
    """dead, home_owns for one period's tracking frame ids."""
    conv = _conv()
    spells = [s for s in conv.recipe_spells(_recipe_input(ev, fps), stop_col="t_stop") if s[0] == period]
    t = frames / fps
    g = ev[ev.period == period]
    dead = frames < g.frame.min()
    dead |= frames > g[["frame", "frame_end"]].max().max()
    for _, s, e, _ in spells:
        dead |= (t >= s) & (t < e)
    own = g[g.type.isin(OWNER_TYPES) & g.team.isin(["home", "away"])]
    idx = np.searchsorted(own.frame.to_numpy(), frames, side="right") - 1
    team = own.team.to_numpy()[np.clip(idx, 0, None)]
    return dead, team == "home"


def _build(g):
    from kloppy import metrica

    if g in (1, 2):
        base = os.path.join(METRICA, f"Sample_Game_{g}_RawTrackingData_")
        ds = metrica.load_tracking_csv(home_data=base + "Home_Team.csv", away_data=base + "Away_Team.csv",
                                       coordinates="metrica")
    else:
        ds = metrica.load_tracking_epts(meta_data=os.path.join(METRICA, "Sample_Game_3_metadata.xml"),
                                        raw_data=os.path.join(METRICA, "Sample_Game_3_tracking.txt"),
                                        coordinates="metrica")
    home_team, away_team = ds.metadata.teams
    L, W = PITCH
    dims = ds.metadata.pitch_dimensions
    if getattr(dims, "pitch_length", None) and getattr(dims, "pitch_width", None):
        L, W = float(dims.pitch_length), float(dims.pitch_width)
    fps = float(ds.metadata.frame_rate)
    ev = events(g)
    if g == 3:   # JSON team ids -> home / away by the tracking metadata
        ev["team"] = ev.team.map({home_team.team_id: "home", away_team.team_id: "away"})
    slots = {tm: {pl.player_id: i for i, pl in enumerate(tm.players)} for tm in (home_team, away_team)}

    def xy(c):
        return ((c.x - 0.5) * L, (0.5 - c.y) * W)

    arrays = dict(fps=fps, half_len=L / 2, half_wid=W / 2)
    for period in ds.metadata.periods:
        fr = [f for f in ds.frames if f.period.id == period.id]
        n, p = len(fr), period.id
        ball = np.full((n, 2), np.nan, np.float32)
        pos = {tm: np.full((n, len(slots[tm]), 2), np.nan, np.float32) for tm in slots}
        for k, f in enumerate(fr):
            if f.ball_coordinates is not None and np.isfinite(f.ball_coordinates.x):
                ball[k] = xy(f.ball_coordinates)
            for pl, d in f.players_data.items():
                if d.coordinates is not None and np.isfinite(d.coordinates.x):
                    pos[pl.team][k, slots[pl.team][pl.player_id]] = xy(d.coordinates)
        frames = np.array([f.frame_id for f in fr])
        dead, home_owns = labels(ev, frames, p, fps)
        arrays[f"t{p}"] = frames / fps
        arrays[f"dead{p}"] = dead
        arrays[f"home_owns{p}"] = home_owns
        arrays[f"ball{p}"] = ball
        arrays[f"home{p}"] = pos[home_team]
        arrays[f"away{p}"] = pos[away_team]
        arrays[f"home_gk{p}"] = np.full(n, -1, np.int16)
        arrays[f"away_gk{p}"] = np.full(n, -1, np.int16)
    return arrays


def load_game(g) -> Match:
    path = os.path.join(CACHE, f"metrica_game_{g}.npz")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, **_build(g))
    z = np.load(path)
    ids = sorted(int(k[1:]) for k in z.files if k.startswith("t") and k[1:].isdigit())
    fields = ("t", "dead", "home_owns", "ball", "home", "away", "home_gk", "away_gk")
    periods = [Period(**{f: z[f"{f}{p}"] for f in fields}) for p in ids]
    return Match(f"metrica{g}", float(z["fps"]), float(z["half_len"]), float(z["half_wid"]), periods)
