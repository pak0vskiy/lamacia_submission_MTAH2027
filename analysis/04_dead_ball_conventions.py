"""How do providers mark dead ball? DFL flag vs DFL events vs Metrica events. Aggregates only.

    uv run python analysis/04_dead_ball_conventions.py

Other games only (IDSSE / DFL and Metrica sample data). Nothing from BAR-ATH is read.

DFL (data/external/idsse/, see analysis/03_idsse_probe.py):
  - end convention: for every set piece (restart), the offset from the nearest end of a flagged dead
    spell to the restart event. Near 0 = the flag goes alive on the restart touch.
  - start convention: offset from the last on-ball event before the stoppage to the start of the
    flagged dead spell.
  - frame-level overlap between the flag and an event recipe: D = [last play event, set piece).

Metrica sample games 1-3 (data/external/metrica/): the same event recipe, with the BALL OUT event
used as the stoppage start when present. Compared with DFL by restart type (spell durations),
since Metrica's open data has no dead-ball flag.

Per-match DFL arrays are cached in artifacts/idsse_flags_<match>.npz (local, gitignored).
"""

import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from kloppy import sportec
from kloppy.domain import BallState, EventType, SetPieceQualifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE = os.path.join(ROOT, "data", "external", "idsse")
METRICA = os.path.join(ROOT, "data", "external", "metrica")
CACHE = os.path.join(ROOT, "artifacts")

# Params
MATCH_WINDOW_S = 5.0      # a set piece and a flagged dead-spell end within this are the same restart
MAX_STOPPAGE_S = 300.0    # ignore recipe spells longer than this (half-time artefacts, broken feeds)
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
NON_PLAY = {EventType.SUBSTITUTION, EventType.CARD, EventType.PLAYER_ON, EventType.PLAYER_OFF,
            EventType.FORMATION_CHANGE}
DFL_NON_PLAY = {"Delete", "FairPlay", "PenaltyNotAwarded", "RefereeBall"}   # generic event names
METRICA_NON_PLAY = {"CARD", "BALL OUT"}


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))   # inclusive (start, end)


def q(values):
    return " / ".join(f"{v:+.2f}" if np.min(values) < 0 else f"{v:.1f}"
                      for v in np.quantile(values, QUANTILES)) if len(values) else "-"


def set_piece_type(event):
    try:
        value = event.get_qualifier_value(SetPieceQualifier)
    except Exception:
        return None
    return value.name if value is not None else None


# ---------------------------------------------------------------- DFL

def dfl_files(match_id):
    pos = glob.glob(os.path.join(IDSSE, f"DFL_04_03_positions_raw_observed_*{match_id}.xml"))[0]
    tag = pos.split("positions_raw_observed_")[1]
    return (pos, os.path.join(IDSSE, f"DFL_02_01_matchinformation_{tag}"),
            os.path.join(IDSSE, f"DFL_03_02_events_raw_{tag}"))


def dfl_flags(match_id):
    """Per period: frame timestamps (s) and dead flag. Cached."""
    path = os.path.join(CACHE, f"idsse_flags_{match_id}.npz")
    if not os.path.exists(path):
        pos, meta, _ = dfl_files(match_id)
        ds = sportec.load_tracking(raw_data=pos, meta_data=meta, only_alive=False)
        arrays = {}
        for period in ds.metadata.periods:
            frames = [f for f in ds.frames if f.period.id == period.id]
            arrays[f"t{period.id}"] = np.array([f.timestamp.total_seconds() for f in frames])
            arrays[f"d{period.id}"] = np.array([f.ball_state == BallState.DEAD for f in frames])
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, fps=ds.metadata.frame_rate, **arrays)
        del ds
    z = np.load(path)
    return float(z["fps"]), {int(k[1:]): (z[k], z[f"d{k[1:]}"]) for k in z.files if k.startswith("t")}


def dfl_events(match_id):
    _, meta, events = dfl_files(match_id)
    ds = sportec.load_event(event_data=events, meta_data=meta)
    rows = []
    for e in ds.events:
        name = getattr(e, "event_name", None)
        goal = e.event_type == EventType.SHOT and getattr(e.result, "name", "") == "GOAL"
        rows.append(dict(period=e.period.id, t=e.timestamp.total_seconds(), kind=e.event_type.name,
                         play=e.event_type not in NON_PLAY and name not in DFL_NON_PLAY,
                         trigger=e.event_type.name in ("BALL_OUT", "FOUL_COMMITTED")
                         or name == "Offside" or goal,
                         set_piece=set_piece_type(e)))
    return pd.DataFrame(rows).sort_values(["period", "t"], kind="stable").reset_index(drop=True)


def recipe_spells(ev, stop_col="t"):
    """D = [stoppage start, set piece). Stoppage start = the latest trigger event (ball out, foul,
    offside, goal) since the previous set piece, else the last play event. Returns
    (period, start, end, type)."""
    out = []
    for period, g in ev.groupby("period"):
        g = g.reset_index(drop=True)
        sp_idx = np.flatnonzero(g.set_piece.notna().to_numpy())
        for k, i in enumerate(sp_idx):
            lo = sp_idx[k - 1] + 1 if k else 0
            before = g.iloc[lo:i]
            before = before[before.t < g.t[i]]
            triggers = before[before.trigger]
            play = before[before.play & before.set_piece.isna()]
            if len(triggers):
                start = triggers.t.iloc[-1]
            elif len(play):
                start = play[stop_col].iloc[-1]
            else:
                continue   # period-opening kick-off, or back-to-back set pieces
            if 0 < g.t[i] - start <= MAX_STOPPAGE_S:
                out.append((period, start, g.t[i], g.set_piece[i]))
    return out


def dfl_report(match_ids):
    end_off, start_off, by_type, recipe_by_type = [], [], defaultdict(list), defaultdict(list)
    n_sp = n_matched = n_spells = n_spells_matched = 0
    inter = union = recipe_total = frames_total = 0
    shares = []
    kinds = Counter()
    for m in match_ids:
        fps, flags = dfl_flags(m)
        ev = dfl_events(m)
        kinds.update(ev.kind)
        spells = recipe_spells(ev)
        dead_total = sum(d.sum() for _, d in flags.values())
        shares.append(dead_total / sum(len(d) for _, d in flags.values()))
        for period, (t, dead) in flags.items():
            fl = [(t[s], t[e] + 1 / fps) for s, e in runs(dead)]
            n_spells += len(fl)
            ends = np.array([e for _, e in fl])
            starts = np.array([s for s, _ in fl])
            sp = [s for s in spells if s[0] == period]
            used = set()
            for _, r_start, r_end, kind in sp:
                n_sp += 1
                recipe_by_type[kind].append(r_end - r_start)
                if not len(ends):
                    continue
                j = int(np.argmin(np.abs(ends - r_end)))
                if abs(ends[j] - r_end) <= MATCH_WINDOW_S:
                    n_matched += 1
                    used.add(j)
                    end_off.append(ends[j] - r_end)
                    start_off.append(starts[j] - r_start)
                    by_type[kind].append(ends[j] - starts[j])
            n_spells_matched += len(used)
            recipe = np.zeros(len(t), bool)
            for _, r_start, r_end, _ in sp:
                recipe |= (t >= r_start) & (t < r_end)
            inter += int((recipe & dead).sum())
            union += int((recipe | dead).sum())
            recipe_total += int(recipe.sum())
            frames_total += len(t)

    print(f"DFL, {len(match_ids)} matches")
    print(f"  flag dead share per match: {', '.join(f'{s:.1%}' for s in shares)}")
    print(f"  event kinds: {dict(kinds.most_common(12))}")
    print(f"  set pieces after a stoppage: {n_sp}; matched to a flagged dead-spell end within "
          f"{MATCH_WINDOW_S:.0f} s: {n_matched} ({n_matched / max(n_sp, 1):.1%})")
    print(f"  flagged dead spells: {n_spells}; matched to a set piece: {n_spells_matched} "
          f"({n_spells_matched / max(n_spells, 1):.1%})")
    print(f"  END offset, flag dead-end minus set-piece time, s, q10/25/50/75/90 = {q(end_off)}")
    print(f"    |offset| <= 0.2 s: {np.mean(np.abs(end_off) <= .2):.1%}, <= 1 s: "
          f"{np.mean(np.abs(end_off) <= 1):.1%}")
    print(f"  START offset, flag dead-start minus event stoppage start (trigger, else last play event), s, q10/25/50/75/90 = {q(start_off)}")
    print(f"    |offset| <= 1 s: {np.mean(np.abs(start_off) <= 1):.1%}, <= 2 s: "
          f"{np.mean(np.abs(start_off) <= 2):.1%}")
    print(f"  event recipe dead share (all matches): {recipe_total / max(frames_total, 1):.1%}")
    print(f"  frame IoU, flag D vs event recipe D: {inter / max(union, 1):.1%}")
    return by_type, recipe_by_type


# ---------------------------------------------------------------- Metrica

def metrica_events():
    games = {}
    for g in (1, 2):
        df = pd.read_csv(os.path.join(METRICA, f"Sample_Game_{g}_RawEventsData.csv"))
        games[g] = pd.DataFrame(dict(period=df["Period"], t=df["Start Time [s]"],
                                     t_end=df["End Time [s]"], type=df["Type"],
                                     sub=df["Subtype"].fillna("")))
    raw = json.load(open(os.path.join(METRICA, "Sample_Game_3_events.json")))["data"]

    def sub_name(s):
        if not s:
            return ""
        return " ".join(x["name"] for x in s) if isinstance(s, list) else s["name"]

    games[3] = pd.DataFrame([dict(period=e["period"], t=e["start"]["time"], t_end=e["end"]["time"],
                                  type=e["type"]["name"], sub=sub_name(e["subtypes"]))
                             for e in raw])
    return games


def metrica_report(dfl_by_type, dfl_recipe_by_type):
    by_type = defaultdict(list)
    shares, n_out = [], 0
    for g, ev in metrica_events().items():
        ev = ev.sort_values(["period", "t"], kind="stable").reset_index(drop=True)
        # goal kicks are tagged as a subtype on the PASS / BALL LOST / BALL OUT that takes them
        goal_kick = ev["sub"].str.startswith("GOAL KICK")
        ev["set_piece"] = np.where(ev.type == "SET PIECE", ev["sub"].str.replace(" ", "_"),
                                   np.where(goal_kick, "GOAL_KICK", None))
        ev["set_piece"] = ev.set_piece.where(ev.set_piece.notna(), None)
        ev["play"] = ~ev.type.isin(METRICA_NON_PLAY)
        ev["trigger"] = ((ev.type == "BALL OUT") & ~goal_kick) | (ev.type == "FAULT RECEIVED") \
            | ev["sub"].str.contains("OFFSIDE|GOAL$|-GOAL", regex=True)
        n_out += int(((ev.type == "BALL OUT") & ~goal_kick).sum())
        # without a trigger, the stoppage starts when the last play event completes
        ev["t_stop"] = ev[["t", "t_end"]].max(axis=1)
        spells = recipe_spells(ev, stop_col="t_stop")
        span = sum(grp.t.max() - grp.t.min() for _, grp in ev.groupby("period"))
        shares.append(sum(e - s for _, s, e, _ in spells) / span)
        for _, s, e, kind in spells:
            by_type[kind].append(e - s)

    print("\nMetrica sample games 1-3, event recipe (no flag available)")
    print(f"  recipe dead share per game: {', '.join(f'{s:.1%}' for s in shares)} "
          f"(between first and last event; excludes stoppages without a set piece)")
    print(f"  stoppages starting at a BALL OUT event: {n_out}")

    print("\nStoppage duration by restart type, s, median [q25-q75] (n)")
    print(f"  {'type':<14} {'DFL flag':>26} {'DFL event recipe':>26} {'Metrica event recipe':>26}")
    names = {"THROW_IN": "THROW_IN", "CORNER_KICK": "CORNER_KICK", "FREE_KICK": "FREE_KICK",
             "GOAL_KICK": "GOAL_KICK", "KICK_OFF": "KICK_OFF", "PENALTY": "PENALTY"}

    def cell(v):
        if not len(v):
            return "-"
        a, b, c = np.quantile(v, [.25, .5, .75])
        return f"{b:5.1f} [{a:4.1f}-{c:5.1f}] ({len(v)})"

    for kind in names:
        print(f"  {kind:<14} {cell(dfl_by_type.get(kind, [])):>26} "
              f"{cell(dfl_recipe_by_type.get(kind, [])):>26} {cell(by_type.get(kind, [])):>26}")


def main():
    ids = sorted({os.path.basename(p).split("DFL-MAT-")[1][:-4]
                  for p in glob.glob(os.path.join(IDSSE, "DFL_04_03_positions_raw_observed_*.xml"))})
    metrica_report(*dfl_report(ids))


if __name__ == "__main__":
    main()
