"""IDSSE probe on all 7 matches + start-of-D convention by stoppage cause. Aggregates only.

    uv run python analysis/05_idsse_all_matches.py

Other games only (IDSSE / DFL). Nothing from BAR-ATH is read.

Part 1 repeats analysis/03_idsse_probe.py (which only ran on J03WPY) for every match and pooled.

Part 2 tests the team's convention hypothesis (general rule, not tied to any BAR-ATH moment):
  - ball out of play: D starts the moment the ball's ground projection leaves the pitch;
  - foul: D starts at the referee's whistle (the DFL foul event time is our whistle proxy).
For every flagged DFL dead spell we look for (a) a ball-exit frame from tracking (ball centre beyond
a touch/goal line; geometric, so no human timing noise) and (b) a foul / offside event near the
dead start, and report flag dead-start minus that anchor.

Per-match arrays are cached in artifacts/idsse_frames_<match>.npz (local, gitignored).
"""

import glob
import importlib.util
import os
from collections import defaultdict

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState, EventType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE = os.path.join(ROOT, "data", "external", "idsse")
CACHE = os.path.join(ROOT, "artifacts")

_spec = importlib.util.spec_from_file_location(
    "conv", os.path.join(ROOT, "analysis", "04_dead_ball_conventions.py"))
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

# Params
EXIT_SEARCH_S = (-6.0, 3.0)   # ball exit must fall in [dead_start - 6 s, dead_start + 3 s]
FOUL_SEARCH_S = (-6.0, 3.0)   # same window for foul / offside events
BALL_RADIUS_M = 0.11          # "fully over the line" = centre beyond line by one radius
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def q(v, signed=False):
    if not len(v):
        return "-"
    fmt = "{:+.2f}" if signed else "{:.1f}"
    return " / ".join(fmt.format(x) for x in np.quantile(v, QUANTILES))


def runs(mask):
    return conv.runs(mask)


def frames(match_id):
    """Per period dict of arrays: t, dead, home_owns, bx, by, bz, ball_ok, n_players. Cached."""
    path = os.path.join(CACHE, f"idsse_frames_{match_id}.npz")
    if not os.path.exists(path):
        pos, meta, _ = conv.dfl_files(match_id)
        ds = sportec.load_tracking(raw_data=pos, meta_data=meta, only_alive=False,
                                   coordinates="sportec")
        home = ds.metadata.teams[0]
        pd_ = ds.metadata.pitch_dimensions
        arrays = dict(fps=ds.metadata.frame_rate, half_len=pd_.x_dim.max, half_wid=pd_.y_dim.max)
        for period in ds.metadata.periods:
            fr = [f for f in ds.frames if f.period.id == period.id]
            b = [f.ball_coordinates for f in fr]
            p = period.id
            arrays[f"t{p}"] = np.array([f.timestamp.total_seconds() for f in fr])
            arrays[f"dead{p}"] = np.array([f.ball_state == BallState.DEAD for f in fr])
            arrays[f"home{p}"] = np.array([f.ball_owning_team == home for f in fr])
            arrays[f"bx{p}"] = np.array([c.x if c is not None else np.nan for c in b], float)
            arrays[f"by{p}"] = np.array([c.y if c is not None else np.nan for c in b], float)
            arrays[f"bz{p}"] = np.array([(c.z or 0.0) if c is not None else np.nan for c in b], float)
            arrays[f"np{p}"] = np.array([len(f.players_data) for f in fr])
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, **arrays)
        del ds
    z = np.load(path)
    periods = sorted(int(k[1:]) for k in z.files if k.startswith("t"))
    out = {p: {k: z[f"{k}{p}"] for k in ("t", "dead", "home", "bx", "by", "bz", "np")} for p in periods}
    return float(z["fps"]), float(z["half_len"]), float(z["half_wid"]), out


# ---------------------------------------------------------------- part 1

def probe(match_id, fps, per):
    r = defaultdict(list)
    for p, a in per.items():
        dead, alive = a["dead"], ~a["dead"]
        r["n"].append(len(dead))
        r["dead"].append(int(dead.sum()))
        r["dead_spells"] += [(e - s + 1) / fps for s, e in runs(dead)]
        r["alive_spells"] += [(e - s + 1) / fps for s, e in runs(alive)]
        both = alive[1:] & alive[:-1]
        r["flips"].append(int(np.sum(a["home"][1:][both] != a["home"][:-1][both])))
        r["alive_min"].append(alive.sum() / fps / 60)
        r["home_alive"].append(int(a["home"][alive].sum()))
        r["n_alive"].append(int(alive.sum()))
        r["ball_ok"].append(int(np.isfinite(a["bx"]).sum()))
        r["z_nz"].append(int((np.nan_to_num(a["bz"]) != 0).sum()))
        r["short"].append(sum(1 for s, e in runs(dead) if (e - s + 1) / fps < 1))
        r["np"].append(a["np"])
    return r


def print_probe(results):
    print("PART 1 - probe per match (analysis/03 numbers, now on all matches)")
    print(f"  {'match':<8} {'min':>5} {'dead':>6} {'spells':>6} {'<1s':>4} "
          f"{'dead spell s q10/25/50/75/90':>32} {'alive spell s q10/25/50/75/90':>32} "
          f"{'flips/alive-h':>13} {'ball':>6} {'z!=0':>6}")
    pooled = defaultdict(list)
    for m, r in results.items():
        n = sum(r["n"])
        print(f"  {m:<8} {n / 25 / 60:5.1f} {sum(r['dead']) / n:6.1%} {len(r['dead_spells']):6d} "
              f"{sum(r['short']):4d} {q(r['dead_spells']):>32} {q(r['alive_spells']):>32} "
              f"{sum(r['flips']) / (sum(r['alive_min']) / 60):13.0f} "
              f"{sum(r['ball_ok']) / n:6.1%} {sum(r['z_nz']) / n:6.1%}")
        for k, v in r.items():
            pooled[k] += v if isinstance(v, list) else [v]
    n = sum(pooled["n"])
    print(f"  {'ALL':<8} {n / 25 / 60:5.1f} {sum(pooled['dead']) / n:6.1%} "
          f"{len(pooled['dead_spells']):6d} {sum(pooled['short']):4d} {q(pooled['dead_spells']):>32} "
          f"{q(pooled['alive_spells']):>32} "
          f"{sum(pooled['flips']) / (sum(pooled['alive_min']) / 60):13.0f} "
          f"{sum(pooled['ball_ok']) / n:6.1%} {sum(pooled['z_nz']) / n:6.1%}")
    nps = np.concatenate(pooled["np"])
    print(f"  players per frame (all) q10/50/90 = {q(nps)}; home share of alive frames "
          f"{sum(pooled['home_alive']) / sum(pooled['n_alive']):.1%}")


# ---------------------------------------------------------------- part 2

def outside(bx, by, hl, hw, margin):
    return (np.abs(bx) > hl + margin) | (np.abs(by) > hw + margin)


def convention(match_id, fps, hl, hw, per, acc):
    ev = conv.dfl_events(match_id)
    fouls = ev[ev.trigger & ~ev.kind.isin(["BALL_OUT", "SHOT"])]   # foul committed, offside
    for p, a in per.items():
        t, dead, bx, by = a["t"], a["dead"], a["bx"], a["by"]
        fp = fouls[fouls.period == p].t.to_numpy()
        for margin_key, margin in (("centre", 0.0), ("full", BALL_RADIUS_M)):
            out = outside(bx, by, hl, hw, margin)
            # exit frame = first frame of an outside run (previous frame inside)
            exits = np.flatnonzero(out[1:] & ~out[:-1]) + 1
            acc[f"exits_{margin_key}"] += len(exits)
            for s, e in runs(dead):
                if s == 0:
                    continue
                t0 = t[s]
                lo, hi = t0 + EXIT_SEARCH_S[0], t0 + EXIT_SEARCH_S[1]
                cand = exits[(t[exits] >= lo) & (t[exits] <= hi)]
                if len(cand):
                    j = cand[np.argmin(np.abs(t[cand] - t0))]
                    acc[f"exit_off_{margin_key}"].append(t0 - t[j])
                    if margin_key == "centre":
                        acc["spell_cause"].append("out")
                        # where did it leave: touchline vs goal line
                        side = "goal line" if abs(bx[j]) > hl else "touchline"
                        acc[f"exit_off_{side}"].append(t0 - t[j])
                        # ball tracked through the exit? (no gap in the 1 s before)
                        w = slice(max(0, j - int(fps)), j)
                        acc["exit_tracked"].append(bool(np.isfinite(bx[w]).all()))
                elif margin_key == "centre":
                    cf = fp[(fp >= t0 + FOUL_SEARCH_S[0]) & (fp <= t0 + FOUL_SEARCH_S[1])]
                    if len(cf):
                        acc["spell_cause"].append("foul")
                        acc["foul_off"].append(t0 - cf[np.argmin(np.abs(cf - t0))])
                    else:
                        acc["spell_cause"].append("other")
            # alive frames with the ball outside: how long after exit does the flag catch up?
            if margin_key == "centre":
                ok = np.isfinite(bx)
                acc["out_frames"] += int((out & ok).sum())
                acc["out_and_alive"] += int((out & ok & ~dead).sum())
                acc["in_frames"] += int((~out & ok).sum())


def print_convention(acc):
    print("\nPART 2 - start of D vs cause (offset = flag dead-start minus anchor; + = flag later)")
    causes = acc["spell_cause"]
    n = len(causes)
    for c in ("out", "foul", "other"):
        k = causes.count(c)
        print(f"  dead spells caused by {c:<5}: {k:4d} ({k / max(n, 1):.1%})")
    for key, label in (("exit_off_centre", "ball centre crosses line"),
                       ("exit_off_full", "ball fully over line (+0.11 m)"),
                       ("exit_off_touchline", "  of which touchline (centre)"),
                       ("exit_off_goal line", "  of which goal line (centre)"),
                       ("foul_off", "foul / offside event (whistle proxy)")):
        v = np.array(acc[key])
        if not len(v):
            continue
        print(f"  {label:<38} n={len(v):4d}  q10/25/50/75/90 = {q(v, True)}  "
              f"|off|<=0.2s {np.mean(np.abs(v) <= .2):5.1%}  <=0.5s {np.mean(np.abs(v) <= .5):5.1%}  "
              f"<=1s {np.mean(np.abs(v) <= 1):5.1%}")
    tr = acc["exit_tracked"]
    print(f"  ball tracked through the full 1 s before exit: {np.mean(tr):.1%} of out-caused spells")
    print(f"  frames with ball outside pitch but flag still alive: "
          f"{acc['out_and_alive'] / max(acc['out_frames'], 1):.1%} of {acc['out_frames']} outside frames")


def main():
    ids = sorted({os.path.basename(p).split("DFL-MAT-")[1][:-4]
                  for p in glob.glob(os.path.join(IDSSE, "DFL_04_03_positions_raw_observed_*.xml"))})
    results, acc = {}, defaultdict(list)
    for k in ("out_frames", "out_and_alive", "in_frames", "exits_centre", "exits_full"):
        acc[k] = 0
    per_match_out = {}
    for m in ids:
        fps, hl, hw, per = frames(m)
        results[m] = probe(m, fps, per)
        before = len(acc["exit_off_centre"])
        convention(m, fps, hl, hw, per, acc)
        v = np.array(acc["exit_off_centre"][before:])
        per_match_out[m] = (len(v), np.median(v) if len(v) else np.nan)
    print_probe(results)
    print_convention(acc)
    print("\n  per match, out-caused spells: n, median offset vs ball-centre exit (s)")
    for m, (k, med) in per_match_out.items():
        print(f"    {m}: n={k:3d}, median {med:+.2f}")


if __name__ == "__main__":
    main()
