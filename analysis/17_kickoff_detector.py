"""Kickoff detector (dead_ball/kickoff.py): accuracy on other games, plausibility on BAR-ATH.

    uv run python analysis/17_kickoff_detector.py

1. Every kickoff in IDSSE (7 matches; DFL flag dead-end at the KICK_OFF event = truth; players
   degraded like our ATD) and Metrica sample games 1-3 (KICK OFF event frame = truth). The detector
   runs from the start of the stoppage that ends at the kickoff (or the period start) to 60 s
   after it. Reported: detection rate and timing error quantiles.
2. False formations in live play: separated runs >= MIN_FORM_S in IDSSE flag-live frames that are
   not within 10 s of a kickoff.
3. BAR-ATH, no labels: the detector on each graded window from its start. Cross-check: does the
   detected kick fall inside Smart Tagging's automatic `Kick Off (Start)` interval for that half?
   The 2nd-half kick frame in FAQ.md is shown as sanity only; nothing is tuned on it.
Aggregates only.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from analysis.atd_cache import GRADED, graded_periods  # noqa: E402
from dead_ball import cv, kickoff  # noqa: E402
from dead_ball.degrade import _drop_players  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.metrica import GAMES, _conv, events, load_game  # noqa: E402

AFTER_S = 60.0
NEAR_S = 10.0
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def dead_run_start(dead, i):
    """Start of the dead run that ends right before frame i (or i itself if none)."""
    j = i
    while j > 0 and dead[j - 1]:
        j -= 1
    return j


def score(cases, lag_s=kickoff.LAG_S):
    errs, missed = [], 0
    cases = [c for c in cases if c[4] - c[3] >= 5 * c[2]]   # testable: >= 5 s of stoppage before the kick
    for home, away, fps, lo, truth, hi in cases:
        k = kickoff.detect(home, away, fps, lo, hi, lag_s=lag_s)
        if k is None:
            missed += 1
        else:
            errs.append((k - truth) / fps)
    errs = np.array(errs)
    return len(cases), missed, errs


def report(name, n, missed, errs):
    a = np.abs(errs)
    print(f"{name}: {n} testable kickoffs (>= 5 s of stoppage tracked before the kick), detected {n - missed}, missed {missed}")
    if len(errs):
        print(f"  error (detected - true), s, q10/25/50/75/90: "
              + " / ".join(f"{x:+.2f}" for x in np.quantile(errs, QS)))
        print(f"  |error| <= 0.5 s {np.mean(a <= .5):.0%}, <= 1 s {np.mean(a <= 1):.0%}, "
              f"<= 3 s {np.mean(a <= 3):.0%}, max {a.max():.1f} s")


def main():
    warnings.simplefilter("ignore")
    conv = _conv()

    # ---- IDSSE
    cases, live_frames, false_runs, starts = [], 0, 0, []
    for i, mid in enumerate(cv.default_ids()):
        m = load_match(mid)
        ev = conv.dfl_events(mid)
        rng = np.random.default_rng([cv.SEED, i, 500])
        for k, p in enumerate(m.periods, start=1):
            home, away = _drop_players(p.home, m.fps, rng, 0.02), _drop_players(p.away, m.fps, rng, 0.02)
            ends = np.flatnonzero(p.dead[:-1] & ~p.dead[1:]) + 1          # first alive frame
            kicks = []
            for t in ev[(ev.period == k) & (ev.set_piece == "KICK_OFF")].t:
                i0 = int(np.searchsorted(p.t, t))
                if i0 < 5 * m.fps and not p.dead[: max(i0, 1)].any():
                    truth = 0                                             # period opens on the kick
                else:
                    near = ends[np.abs(ends - i0) <= 5 * m.fps]
                    if not len(near):
                        continue
                    truth = int(near[np.argmin(np.abs(near - i0))])
                kicks.append(truth)
                lo = dead_run_start(p.dead, truth)
                cases.append((home, away, m.fps, lo, truth, min(len(p), truth + int(AFTER_S * m.fps))))
                if truth == 0:
                    starts.append(len(cases) - 1)
            sep = kickoff.separated(home, away)
            live = ~p.dead
            live_frames += int(live.sum())
            for s, e in kickoff._runs(sep & live):
                if e - s >= kickoff.MIN_FORM_S * m.fps and not any(abs(s - t) < NEAR_S * m.fps or abs(e - t) < NEAR_S * m.fps for t in kicks):
                    false_runs += 1
        del m
    idsse_cases = cases
    n, missed, errs = score(cases)
    report("IDSSE (all kickoffs)", n, missed, errs)
    print(f"  not testable: {len(idsse_cases) - n} (tracking starts on the kick)")
    print(f"  false formations in live play (>= {kickoff.MIN_FORM_S:g} s, > {NEAR_S:g} s from a kickoff): "
          f"{false_runs} in {live_frames / 25 / 60:.0f} live min")

    # ---- Metrica
    cases = []
    for g in GAMES:
        m = load_game(g)
        ev = events(g)
        for k, p in enumerate(m.periods, start=1):
            frames = np.round(p.t * m.fps).astype(int)
            for f in ev[(ev.period == k) & (ev.type == "SET PIECE") & (ev["sub"] == "KICK OFF")].frame:
                truth = int(np.searchsorted(frames, f))
                lo = dead_run_start(p.dead, truth)
                cases.append((p.home, p.away, m.fps, lo, truth, min(len(p), truth + int(AFTER_S * m.fps))))
        del m
    report("\nMetrica (all kickoffs)", *score(cases))
    raw = np.r_[score(idsse_cases, 0.0)[2], score(cases, 0.0)[2]]
    print(f"\npooled median raw lag (no correction), IDSSE + Metrica: {np.median(raw):+.2f} s "
          f"(LAG_S = {kickoff.LAG_S})")

    # ---- BAR-ATH
    import load_smart_tagging as st
    tags = [e for e in st.load()
            if (e["code"] or "").strip().lower() == "set pieces"
            and (e["tags"].get("Type") or "").strip().lower() == "kick off (start)"]
    periods, _, _ = graded_periods()
    print("\nBAR-ATH, label-free:")
    for h, ((s, e), p) in enumerate(zip(GRADED, periods), start=1):
        k = kickoff.detect(p.home, p.away, 25.0, 0, len(p))
        if k is None:
            print(f"  half {h}: no kickoff detected")
            continue
        half = "1st Half" if h == 1 else "2nd Half"
        inside = [t for t in tags if t["tags"].get("Half") == half and t["start_frame"] <= s + k <= t["end_frame"]]
        print(f"  half {h}: kick detected {k / 25:.1f} s after the graded start; "
              f"inside Smart Tagging's Kick Off (Start) interval: {bool(inside)}")
        if h == 2:
            print(f"    sanity only (FAQ.md kick frame): detected - FAQ = {(s + k - 113742) / 25:+.2f} s")


if __name__ == "__main__":
    main()
