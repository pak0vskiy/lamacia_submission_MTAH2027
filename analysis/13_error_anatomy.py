"""Where does the team logic lose spellF1? Error anatomy + boundary timing on held-out IDSSE.

    uv run python analysis/13_error_anatomy.py [--matches N]

Same held-out copies, D fold models and ball smoothing as analysis/09 (calibrated blackouts),
team logic = Vidal-Codina TUNED. Other games only; nothing from BAR-ATH is read. Aggregates only.

Three read-outs:
1. Error anatomy at IoU 0.5: every missed true possession and every spurious predicted one gets
   one cause (merged / split / other team holds / predicted dead / boundary ...), plus recall by
   possession length and by whether the ball was hidden at the possession's start.
2. Boundary timing: signed offset (pred - truth, frames) of D starts, D ends and H<->A switches,
   matched one-to-one within +/- 2 s.
3. Shift sweeps: spellF1 when D starts, D ends or team switches are moved by k frames. The value
   chosen for each held-out match is the best on the other matches (leave-one-match-out).
"""

import argparse
import os
import sys
import warnings
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from dead_ball import cv  # noqa: E402
from dead_ball.degrade import degrade, smooth_ball  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.pipeline import A, D, H, X, decode_dead, oracle_codes, truth_codes  # noqa: E402
from possession.vidal_codina import TUNED, player_possession, team_codes  # noqa: E402

MODEL = "model-no-ball_spd_1@mixed+cal"
BALL_SMOOTH_S = 0.25
IOU = 0.5
TOL = 50                       # frames (2 s) for transition matching
HIDDEN_WIN = 12                # +/- frames around a possession start for "ball hidden"
LEN_BINS = (0, 2, 5, 10, 20, 1e9)
SHIFTS_D = (-25, -12, -6, -3, 0, 3, 6, 12, 25)
SHIFTS_SW = (-12, -6, -3, 0, 3, 6, 12)
OUT = os.path.join(cv.OUT, "..", "possession")


# ---------------------------------------------------------------- helpers

def rebuild(m, i, j, c):
    rng = np.random.default_rng([cv.SEED, i, 100 + j])
    return [degrade(p, m.fps, rng, c, motion=cv.DEFAULT_MOTION, blackout=cv.DEFAULT_BLACKOUT)
            for p in m.periods]


def runs_of(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def shift_dead(dead, s_start, s_end):
    out = np.zeros_like(dead)
    n = len(dead)
    for a, b in runs_of(dead):
        a2 = a if a == 0 else min(max(a + s_start, 0), n)          # don't move edges at period ends
        b2 = b if b == n else min(max(b + s_end, 0), n)
        if b2 > a2:
            out[a2:b2] = True
    return out


def shift_switches(codes, k):
    """Move every direct H<->A change by k frames (k > 0 later), never across a D."""
    out = codes.copy()
    if k == 0:
        return out
    t = np.flatnonzero((codes[1:] != codes[:-1]) & np.isin(codes[1:], (H, A)) & np.isin(codes[:-1], (H, A))) + 1
    segs = pm.segments(codes, [(0, len(codes))])
    starts = {s: (a, b) for a, b, _ in segs for s in [a]}
    ends = {b: (a, b) for a, b, _ in segs}
    for x in t:
        prev_a, _ = ends[x]
        _, next_b = starts[x]
        if k > 0:
            y = min(x + k, next_b - 1)
            out[x:y] = codes[x - 1]
        else:
            y = max(x + k, prev_a + 1)
            out[y:x] = codes[x]
    return out


def filled(codes):
    return pm.fill_abstentions(codes, [(0, len(codes))])


def greedy_match(ov, threshold):
    gm, pmatch = set(), set()
    for iou, g, p in ov:
        if iou < threshold:
            break
        if g in gm or p in pmatch:
            continue
        gm.add(g)
        pmatch.add(p)
    return gm, pmatch


def anatomy(gt, pred, hidden_start):
    """Cause per missed GT possession and per spurious predicted one (IoU 0.5)."""
    runs = [(0, len(gt))]
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    gm, pmatch = greedy_match(pm.spell_overlaps(gs, ps), IOU)
    gt_rows, pred_rows = [], []
    for i, (a, b, k) in enumerate(gs):
        other = A if k == H else H
        seg = pred[a:b]
        dur = (b - a) / 25.0
        if i in gm:
            cause = "matched"
        elif np.mean(seg == D) >= 0.5:
            cause = "pred dead"
        elif np.mean(seg == other) >= 0.5:
            cause = "other team holds"
        else:
            same = [(pa, pb) for pa, pb, pk in ps if pk == k and pb > a and pa < b]
            # best same-team pred spell by overlap
            pa, pb = max(same, key=lambda s: min(s[1], b) - max(s[0], a))
            covers = [j for j, (ga, gb, gk) in enumerate(gs) if gk == k and j != i and gb > pa and ga < pb
                      and (min(gb, pb) - max(ga, pa)) >= 0.5 * (gb - ga)]
            if covers:
                j = covers[0]
                lo, hi = (b, gs[j][0]) if j > i else (gs[j][1], a)
                between = gt[lo:hi]
                cause = "merged across D" if (between == D).any() and not (between == other).any() else \
                        "merged across switch"
            elif len(same) >= 2:
                cause = "split"
            else:
                cause = "boundary"
        gt_rows.append(dict(cause=cause, dur=dur, hidden=bool(hidden_start[i])))
    for j, (a, b, k) in enumerate(ps):
        if j in pmatch:
            continue
        other = A if k == H else H
        seg = gt[a:b]
        if np.mean(seg == D) >= 0.5:
            cause = "inside true D"
        elif np.mean(seg == other) >= 0.5:
            cause = "false flip"
        else:
            within = [(ga, gb) for ga, gb, gk in gs if gk == k and ga <= a + 2 and b - 2 <= gb]
            cause = "fragment of true" if within and (within[0][1] - within[0][0]) > (b - a) * 1.5 else "boundary"
        pred_rows.append(dict(cause=cause, dur=(b - a) / 25.0))
    return gt_rows, pred_rows


def transitions(codes):
    t = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    out = []
    for x in t:
        f, to = codes[x - 1], codes[x]
        if f in (H, A) and to == D:
            out.append((x, "D start"))
        elif f == D and to in (H, A):
            out.append((x, "D end"))
        elif f in (H, A) and to in (H, A):
            out.append((x, "switch"))
    return out


def timing(gt, pred):
    rows = []
    gtr, prr = transitions(gt), transitions(pred)
    for typ in ("D start", "D end", "switch"):
        g = [x for x, t in gtr if t == typ]
        p = [x for x, t in prr if t == typ]
        pairs = sorted((abs(pp - gg), gi, pi, pp - gg) for gi, gg in enumerate(g) for pi, pp in enumerate(p)
                       if abs(pp - gg) <= TOL) if g and p else []
        used_g, used_p = set(), set()
        for _, gi, pi, off in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            rows.append(dict(type=typ, offset=off))
        rows.append(dict(type=typ, n_gt=len(g), n_pred=len(p), n_matched=len(used_g)))
    return rows


def spell_f1(gt_list, pred_list):
    sep = np.array([X], np.int8)
    g = np.concatenate([np.r_[sep, x] if i else x for i, x in enumerate(gt_list)])
    p = np.concatenate([np.r_[sep, x] if i else x for i, x in enumerate(pred_list)])
    runs = pm.valid_runs(g)
    return pm.spell_scores(g, pm.fill_abstentions(p, runs), runs)["mean_f1"]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    ids = cv.default_ids(a.matches)
    data = cv.prepare(ids, 2, cv.DEFAULT_MOTION, cv.DEFAULT_BLACKOUT)
    model = cv.Model(drop=cv.DEFAULT_DROP)

    gt_rows, pred_rows, time_rows, sweep_rows = [], [], [], []
    for i, mid in enumerate(ids):
        fold = joblib.load(os.path.join(cv.OUT, f"fold_{MODEL}_{mid}.joblib"))  # our own fold model
        m = load_match(mid)
        for j, c in enumerate(cv.TEST_COINCIDENCE):
            copy = data[mid]["test"][c]
            qs = rebuild(m, i, j, c)
            for q, f in zip(qs, copy.feats):
                assert np.array_equal(np.isfinite(q.ball[:, 0]), f.ball_vis.to_numpy() > 0)
            per = []
            for q, f in zip(qs, copy.feats):
                vis = np.isfinite(q.ball[:, 0])
                qs_ball = smooth_ball(q.ball, m.fps, BALL_SMOOTH_S)
                players = np.concatenate([q.home, q.away], axis=1)
                team = np.r_[np.zeros(q.home.shape[1], int), np.ones(q.away.shape[1], int)]
                owner = player_possession(qs_ball, players, m.fps, TUNED)
                dead = decode_dead(model.predict(fold["fitted"], f), m.fps, **fold["params"])
                per.append(dict(q=q, vis=vis, owner=owner, team=team, dead=dead, gt=truth_codes(q)))

            tag = dict(match=mid, copy=c)
            for dmode in ("model", "oracle"):
                for x in per:
                    d = x["dead"] if dmode == "model" else x["q"].dead
                    pred = filled(team_codes(x["owner"], x["team"], d))
                    gs = pm.possession_spells(x["gt"], [(0, len(x["gt"]))])
                    n = len(x["vis"])
                    hidden = [np.mean(~x["vis"][max(s - HIDDEN_WIN, 0):min(s + HIDDEN_WIN + 1, n)]) > 0.5
                              for s, _, _ in gs]
                    g_r, p_r = anatomy(x["gt"], pred, hidden)
                    gt_rows += [dict(r, d=dmode, method="vc", **tag) for r in g_r]
                    pred_rows += [dict(r, d=dmode, method="vc", **tag) for r in p_r]
                    time_rows += [dict(r, d=dmode, method="vc", **tag) for r in timing(x["gt"], pred)]
                    if dmode == "model":   # D edges with DFL's owner: the D model's own timing
                        po = filled(oracle_codes(x["q"], x["dead"]))
                        time_rows += [dict(r, d=dmode, method="dfl", **tag) for r in timing(x["gt"], po)]

            gts = [x["gt"] for x in per]
            for s in SHIFTS_D:
                for which in ("D start", "D end"):
                    deads = [shift_dead(x["dead"], s if which == "D start" else 0, s if which == "D end" else 0)
                             for x in per]
                    vc = [team_codes(x["owner"], x["team"], d) for x, d in zip(per, deads)]
                    dfl = [oracle_codes(x["q"], d) for x, d in zip(per, deads)]
                    sweep_rows.append(dict(method="vc", sweep=which, shift=s, f1=spell_f1(gts, vc), **tag))
                    sweep_rows.append(dict(method="dfl", sweep=which, shift=s, f1=spell_f1(gts, dfl), **tag))
            for s in SHIFTS_SW:
                vc = [shift_switches(filled(team_codes(x["owner"], x["team"], x["dead"])), s) for x in per]
                sweep_rows.append(dict(method="vc", sweep="switch", shift=s, f1=spell_f1(gts, vc), **tag))
        print(f"  done {mid}", flush=True)
        del m

    G, P, T, S = (pd.DataFrame(r) for r in (gt_rows, pred_rows, time_rows, sweep_rows))
    os.makedirs(OUT, exist_ok=True)
    for name, df in (("gt", G), ("pred", P), ("timing", T), ("sweep", S)):
        df.to_csv(os.path.join(OUT, f"anatomy_{name}.csv"), index=False)   # local only

    pd.set_option("display.width", 200)
    nm = len(ids) * len(cv.TEST_COINCIDENCE)
    print(f"\n{len(ids)} held-out matches x {len(cv.TEST_COINCIDENCE)} copies. VC TUNED, IoU {IOU}.")

    for dmode in ("model", "oracle"):
        g = G[G.d == dmode]
        p = P[P.d == dmode]
        print(f"\n=== D = {dmode} ===")
        print(f"true possessions per match {len(g) / nm:.0f}, predicted {(len(p) + (g.cause == 'matched').sum()) / nm:.0f}")
        t = g.groupby("cause").agg(n=("dur", "size"), med_dur_s=("dur", "median"))
        t["share"] = t.n / len(g)
        t["per_match"] = t.n / nm
        print("\nTRUE possessions by outcome:")
        print(t.sort_values("n", ascending=False).round(3).to_string())
        t = p.groupby("cause").agg(n=("dur", "size"), med_dur_s=("dur", "median"))
        t["per_match"] = t.n / nm
        print("\nSPURIOUS predicted possessions by cause:")
        print(t.sort_values("n", ascending=False).round(3).to_string())
        g = g.assign(len_bin=pd.cut(g.dur, LEN_BINS, right=False))
        r = g.groupby("len_bin", observed=True).agg(n=("dur", "size"), recall=("cause", lambda s: np.mean(s == "matched")))
        r["share"] = r.n / len(g)
        print("\nrecall @IoU0.5 by true possession length (s):")
        print(r.round(3).to_string())
        r = g.groupby("hidden").agg(n=("dur", "size"), recall=("cause", lambda s: np.mean(s == "matched")))
        print("\nrecall by ball hidden at possession start (>50% of +/-0.5 s):")
        print(r.round(3).to_string())

    print("\n=== Boundary timing: offset = pred - truth (frames, 25 fps), matched within +/-2 s ===")
    off = T[T.offset.notna()]
    cnt = T[T.n_gt.notna()].groupby(["method", "d", "type"])[["n_gt", "n_pred", "n_matched"]].sum()
    q = off.groupby(["method", "d", "type"]).offset.describe(percentiles=[.1, .25, .5, .75, .9])
    q["within5"] = off.groupby(["method", "d", "type"]).offset.apply(lambda s: np.mean(np.abs(s) <= 5))
    q["recall2s"] = cnt.n_matched / cnt.n_gt
    q["pred/gt"] = cnt.n_pred / cnt.n_gt
    print(q[["count", "mean", "10%", "25%", "50%", "75%", "90%", "within5", "recall2s", "pred/gt"]].round(2).to_string())

    print("\n=== Shift sweeps: mean spellF1 over held-out matches (D = model) ===")
    for (method, sweep), s in S.groupby(["method", "sweep"]):
        per_match = s.groupby(["match", "shift"]).f1.mean().unstack("shift")
        base = per_match[0]
        row = " ".join(f"{k:+d}:{v:.3f}" for k, v in per_match.mean().items())
        print(f"{method:4s} {sweep:8s} {row}")
        if len(per_match) < 2:
            continue
        lomo = []
        for mid in per_match.index:
            best = per_match.drop(index=mid).mean().idxmax()
            lomo.append((best, per_match.loc[mid, best] - base[mid]))
        chosen = pd.Series([b for b, _ in lomo]).value_counts().to_dict()
        gain = np.array([d for _, d in lomo])
        print(f"     LOMO choice {chosen}  held-out gain {gain.mean():+.4f} (min {gain.min():+.4f}, max {gain.max():+.4f})")


if __name__ == "__main__":
    main()
