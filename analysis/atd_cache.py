"""Parse the tactical ATD once into numpy arrays, cached in artifacts/ (gitignored).

    from analysis.atd_cache import load
    atd = load()      # dict: px, py, ps (n, slots) in METRES / m/s, ball (n, 3), slot_team, slot_gk, ...

Row i is video frame i (= ATD frame_id - 1). Positions are converted to metres with the
games.py pitch size; NaN where missing. Nothing here prints data.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

# Params
CACHE = os.path.join(ROOT, "artifacts", "atd_tactical.npz")
GRADED = ((11185, 90652), (110808, 190198))   # half-open video frames, competition/DATA.md


def _strip(tag):
    return tag.split("}", 1)[-1]


def _meta(path):
    root = ET.parse(path).getroot()
    team_ids = [t.attrib["id"] for t in root.iter() if _strip(t.tag) == "Team"]
    slot_team, slot_gk = [], []
    attack_first_half = {}
    for node in root.iter():
        tag = _strip(node.tag)
        if tag == "Player":
            slot_team.append(team_ids.index(node.attrib["teamId"]))
            params = {p.findtext("{*}Name"): (p.findtext("{*}Value") or "").strip()
                      for p in node.iter() if _strip(p.tag) == "ProviderParameter"}
            slot_gk.append(params.get("position_type") == "Goalkeeper")
        if tag == "Team":
            for p in node.iter():
                if _strip(p.tag) == "ProviderParameter" and p.findtext("{*}Name") == "attack_direction_first_half":
                    attack_first_half[team_ids.index(node.attrib["id"])] = p.findtext("{*}Value").strip()
    # team index 0 is the first team in the metadata = home (H)
    return np.array(slot_team), np.array(slot_gk), attack_first_half


def _parse(path, n_slots):
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


def load(rebuild=False):
    if not rebuild and os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        out = {k: z[k] for k in z.files}
    else:
        paths = asset_paths()
        slot_team, slot_gk, attack = _meta(paths["atd_meta"])
        px, py, ps, ball = _parse(paths["atd_raw"], len(slot_team))
        length, width = GAMES[GAME]["pitch"]
        out = {
            "px": px * length, "py": py * width, "ps": ps,
            "ball_x": ball[:, 0] * length, "ball_y": ball[:, 1] * width, "ball_s": ball[:, 2],
            "slot_team": slot_team, "slot_gk": slot_gk,
            # +1 if team 0 (home) attacks towards x = length in the 1st half, else -1
            "home_attacks_plus_x_first_half": np.array(attack.get(0) == "left_to_right"),
        }
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **out)

    out["pitch"] = np.array(GAMES[GAME]["pitch"])
    out["halves"] = np.array(half_video_blocks(GAME))   # rows: (half, first, last) video frames
    return out


def graded_periods():
    """BAR-ATH graded windows as dead_ball.idsse.Period objects (centre-origin metres), no labels.

    Returns (periods, half_len, half_wid). `dead` and `home_owns` are all-False placeholders: we
    have no labels for this game and never use any.
    """
    from dead_ball.idsse import Period
    a = load()
    hl, hw = (float(v) / 2 for v in a["pitch"])
    team, gk = a["slot_team"], a["slot_gk"].astype(bool)
    periods = []
    for s, e in GRADED:
        xy = np.stack([a["px"][s:e], a["py"][s:e]], axis=-1).astype(np.float32) - [hl, hw]
        ball = np.stack([a["ball_x"][s:e], a["ball_y"][s:e]], axis=-1).astype(np.float32) - [hl, hw]
        sides = []
        for k in (0, 1):
            pxy = xy[:, team == k]
            g = np.flatnonzero(gk[team == k])
            idx = np.full(len(pxy), -1, np.int16)
            for slot in g:
                idx[np.isfinite(pxy[:, slot, 0]) & (idx < 0)] = slot
            sides += [pxy, idx]
        n = e - s
        periods.append(Period(t=np.arange(n) / 25.0, dead=np.zeros(n, bool),
                              home_owns=np.zeros(n, bool), ball=ball, home=sides[0],
                              away=sides[2], home_gk=sides[1], away_gk=sides[3]))
    return periods, hl, hw
