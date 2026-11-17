"""Do ball blackouts on degraded IDSSE look like BAR-ATH's? Label-free, aggregates only.

    uv run python analysis/12_blackout_calibration.py [--blackout coincidence|calibrated]

Splits every frame into live / dead by our dead-ball model's own calls, on BOTH sides (BAR-ATH:
the submission model, scripts/make_submission.py; IDSSE: the held-out fold model), so the two are
compared like for like. No labels of any kind from BAR-ATH.

Statistics:
- missing share of the ball in live and in dead frames;
- gap length quantiles (s) for live and dead gaps (gap state = majority of its frames);
- live gaps per live minute;
- crowding at the last visible frame before a live gap vs at a random visible live frame:
  players within CROWD_M of the ball, and the share of "duels" (>= 1 player of each team within).
"""

import argparse
import importlib.util
import os
import sys
import warnings

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.atd_cache import graded_periods  # noqa: E402
from dead_ball import cv  # noqa: E402
from dead_ball.features import frame_features  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.pipeline import decode_dead  # noqa: E402

FPS = 25.0
CROWD_M = 3.0
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
FOLD_MODEL = "model-no-ball_spd_1@mixed"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def crowd(p, frames, ball):
    """(players within CROWD_M, both teams within CROWD_M) at `frames`, ball positions `ball`."""
    dh = np.linalg.norm(p.home[frames] - ball[:, None, :], axis=2)
    da = np.linalg.norm(p.away[frames] - ball[:, None, :], axis=2)
    nh = np.sum(np.nan_to_num(dh, nan=np.inf) < CROWD_M, axis=1)
    na = np.sum(np.nan_to_num(da, nan=np.inf) < CROWD_M, axis=1)
    return nh + na, (nh > 0) & (na > 0)


def stats(periods, deads):
    """Aggregate blackout statistics over periods with per-frame predicted dead masks."""
    s = dict(live=0, dead=0, live_miss=0, dead_miss=0, live_gaps=[], dead_gaps=[],
             gap_n=[], gap_duel=[], base_n=[], base_duel=[])
    for p, dead in zip(periods, deads):
        miss = ~np.isfinite(p.ball[:, 0])
        s["live"] += (~dead).sum()
        s["dead"] += dead.sum()
        s["live_miss"] += (miss & ~dead).sum()
        s["dead_miss"] += (miss & dead).sum()
        starts = []
        for a, b in _runs(miss):
            is_dead = dead[a:b].mean() > 0.5
            s["dead_gaps" if is_dead else "live_gaps"].append((b - a) / FPS)
            if not is_dead and a > 0:
                starts.append(a - 1)
        if starts:
            f = np.array(starts)
            n, duel = crowd(p, f, p.ball[f])
            s["gap_n"].append(n)
            s["gap_duel"].append(duel)
        base = np.flatnonzero(~miss & ~dead)[::5]
        n, duel = crowd(p, base, p.ball[base])
        s["base_n"].append(n)
        s["base_duel"].append(duel)
    for k in ("gap_n", "gap_duel", "base_n", "base_duel"):
        s[k] = np.concatenate(s[k]) if s[k] else np.zeros(0)
    return s


FINE = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MIN_SPELL_S = 2.0


def spell_stats(periods, deads):
    """Dead-spell-anchored blackout statistics (the inputs of degrade.calibrated_mask).

    For every dead spell >= MIN_SPELL_S: does a gap overlap it, and for the gap with the largest
    overlap, its start / end offset (s) relative to the spell's start / end. Live-only gaps (not
    touching any dead spell): count, length. Duel share at the frame before a live-only gap vs at a
    random visible live frame."""
    cover, start_off, end_off, live_len, live_min = [], [], [], [], 0.0
    duel_gap, duel_base = [], []
    for p, dead in zip(periods, deads):
        miss = ~np.isfinite(p.ball[:, 0])
        gaps = _runs(miss)
        spells = [(a, b) for a, b in _runs(dead) if b - a >= MIN_SPELL_S * FPS]
        touched = np.zeros(len(gaps), bool)
        for a, b in _runs(dead):
            for k, (g0, g1) in enumerate(gaps):
                if g0 < b and g1 > a:
                    touched[k] = True
        for a, b in spells:
            ov = [(min(b, g1) - max(a, g0), g0, g1) for g0, g1 in gaps if g0 < b and g1 > a]
            cover.append(bool(ov))
            if ov:
                _, g0, g1 = max(ov)
                start_off.append((g0 - a) / FPS)
                end_off.append((g1 - b) / FPS)
        live_min += (~dead).sum() / FPS / 60
        starts = []
        for (g0, g1), t in zip(gaps, touched):
            if not t:
                live_len.append((g1 - g0) / FPS)
                if g0 > 0:
                    starts.append(g0 - 1)
        if starts:
            f = np.array(starts)
            duel_gap.append(crowd(p, f, p.ball[f])[1])
        base = np.flatnonzero(~miss & ~dead)[::5]
        duel_base.append(crowd(p, base, p.ball[base])[1])
    duel_gap = np.concatenate(duel_gap) if duel_gap else np.zeros(0)
    return dict(n_spells=len(cover), p_cover=np.mean(cover),
                start_off_q=np.quantile(start_off, FINE), end_off_q=np.quantile(end_off, FINE),
                n_live_gaps=len(live_len), live_gaps_per_min=len(live_len) / live_min,
                live_len_q=np.quantile(live_len, FINE) if live_len else np.full(len(FINE), np.nan),
                live_duel_ratio=duel_gap.mean() / np.concatenate(duel_base).mean() if len(duel_gap) else np.nan)


def report_spells(name, r):
    q = lambda v: " ".join(f"{x:6.2f}" for x in v)  # noqa: E731
    print(f"  [{name}] dead spells >= {MIN_SPELL_S:.0f} s: {r['n_spells']}, covered by a gap {r['p_cover']:.1%}")
    print(f"    gap start - spell start (s) q{FINE}: {q(r['start_off_q'])}")
    print(f"    gap end   - spell end   (s) q{FINE}: {q(r['end_off_q'])}")
    print(f"    live-only gaps {r['n_live_gaps']} = {r['live_gaps_per_min']:.2f} per live min; "
          f"length s: {q(r['live_len_q'])}; duel ratio {r['live_duel_ratio']:.2f}")


def summarise(s):
    out = dict(
        live_missing=s["live_miss"] / s["live"],
        dead_missing=s["dead_miss"] / s["dead"],
        dead_share=s["dead"] / (s["live"] + s["dead"]),
        live_gaps_per_min=len(s["live_gaps"]) / (s["live"] / FPS / 60),
        live_gap_q=np.quantile(s["live_gaps"], QS) if s["live_gaps"] else np.full(len(QS), np.nan),
        dead_gap_q=np.quantile(s["dead_gaps"], QS) if s["dead_gaps"] else np.full(len(QS), np.nan),
        crowd_gap=s["gap_n"].mean(), crowd_base=s["base_n"].mean(),
        duel_gap=s["gap_duel"].mean(), duel_base=s["base_duel"].mean(),
    )
    return out


def report(name, r):
    q = lambda v: " / ".join(f"{x:5.1f}" for x in v)  # noqa: E731
    print(f"\n{name}")
    print(f"  predicted dead share {r['dead_share']:.1%}; ball missing: live {r['live_missing']:.1%}, "
          f"dead {r['dead_missing']:.1%}")
    print(f"  live gaps {r['live_gaps_per_min']:.2f} per live min; length s q10..q90 {q(r['live_gap_q'])}")
    print(f"  dead gaps length s q10..q90 {q(r['dead_gap_q'])}")
    print(f"  players within {CROWD_M:.0f} m of ball: before a live gap {r['crowd_gap']:.2f} vs any live frame "
          f"{r['crowd_base']:.2f}; duel share {r['duel_gap']:.1%} vs {r['duel_base']:.1%} "
          f"(ratio {r['duel_gap'] / max(r['duel_base'], 1e-9):.2f})")


def bar_ath():
    ms = _load("ms", os.path.join("scripts", "make_submission.py"))
    bundle = ms.train_dead_model()
    model = cv.Model(drop=bundle["drop"])
    periods, hl, hw = graded_periods()
    deads = [decode_dead(model.predict(bundle["fitted"], frame_features(p, FPS, hl, hw)), FPS, **bundle["params"])
             for p in periods]
    return summarise(stats(periods, deads)), spell_stats(periods, deads)


def idsse(coincidences, blackout="coincidence"):
    """Held-out test copies rebuilt from analysis/06's seeds; D from each match's fold model."""
    from dead_ball.degrade import degrade
    out = {}
    ids = cv.default_ids()
    model = cv.Model(drop=cv.DEFAULT_DROP)
    acc = {c: ([], []) for c in coincidences}
    for i, mid in enumerate(ids):
        # joblib = pickle: only our own fold models, written locally by dead_ball/cv.py
        fold = joblib.load(os.path.join(cv.OUT, f"fold_{FOLD_MODEL}_{mid}.joblib"))
        m = load_match(mid)
        for c in coincidences:
            j = cv.TEST_COINCIDENCE.index(c)
            rng = np.random.default_rng([cv.SEED, i, 100 + j])
            qs = [degrade(p, m.fps, rng, c, motion=cv.DEFAULT_MOTION, blackout=blackout) for p in m.periods]
            for q in qs:
                f = frame_features(q, m.fps, m.half_len, m.half_wid)
                acc[c][0].append(q)
                acc[c][1].append(decode_dead(model.predict(fold["fitted"], f), m.fps, **fold["params"]))
        del m
    for c in coincidences:
        out[c] = summarise(stats(*acc[c])), spell_stats(*acc[c])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blackout", default="coincidence", choices=["coincidence", "calibrated"])
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    r, sp = bar_ath()
    report("BAR-ATH graded frames (no labels; D = submission model's calls)", r)
    report_spells("BAR-ATH", sp)
    cs = (0.5, 0.9) if a.blackout == "coincidence" else (0.5,)
    for c, (r, sp) in idsse(cs, a.blackout).items():
        tag = f"c = {c}" if a.blackout == "coincidence" else "calibrated"
        report(f"IDSSE degraded ({tag}; D = held-out fold model's calls)", r)
        report_spells("IDSSE", sp)


if __name__ == "__main__":
    main()
