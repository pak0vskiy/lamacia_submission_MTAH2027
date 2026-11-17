"""What would a restart-team hint (Smart Tagging set pieces) be worth? Held-out IDSSE. Aggregates only.

    uv run python analysis/18_restart_hints.py

Smart Tagging exists only for BAR-ATH, so its value can't be measured there without labels. This
simulates it on other games instead. Same held-out setting as analysis/14 (fold D model, fold owner
model, calibrated blackouts, 0.25 s ball smoothing, 3 test copies per match), switch cost
learned.SWITCH_COST. A "hint" is a true restart (DFL flag dead -> alive) with the team that has
the ball on its first alive frame, spread over a 5 s interval like Smart Tagging's tags. A
predicted live stretch whose first frame lies within HINT_WINDOW_S of a hint starts with that team
(learned.restart_starts / decode(starts=...)).

Variants:
- which restarts are hinted. Smart Tagging tags corners, goal kicks, kickoffs and a few free kicks
  with a team, and throw-ins without one;
- team flipped on a share of hints, as a stand-in for tagging errors.
Other games only; nothing from BAR-ATH is read.
"""

import importlib.util
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dead_ball import cv  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.pipeline import decode_dead, truth_codes  # noqa: E402
from possession import learned  # noqa: E402

spec = importlib.util.spec_from_file_location("a14", os.path.join(ROOT, "analysis", "14_learned_team_cv.py"))
a14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a14)
spec = importlib.util.spec_from_file_location("a04", os.path.join(ROOT, "analysis", "04_dead_ball_conventions.py"))
a04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a04)

# Params (fixed before running)
TAG_HALF_S = 2.5           # hint interval = true restart +- this (Smart Tagging marks ~5 s windows)
HINT_WINDOW_S = 2.0        # a stretch start this far outside the interval still takes the hint
TYPE_MATCH_S = 3.0         # DFL set-piece event this close to the restart gives its type
FLIPS = (0.0, 0.1, 0.25)
SETS = {
    "corner+goal kick+kickoff": {"CORNER_KICK", "GOAL_KICK", "KICK_OFF", "PENALTY"},
    "+ free kicks": {"CORNER_KICK", "GOAL_KICK", "KICK_OFF", "PENALTY", "FREE_KICK"},
    "+ throw-ins (all set pieces)": {"CORNER_KICK", "GOAL_KICK", "KICK_OFF", "PENALTY", "FREE_KICK", "THROW_IN"},
}
CACHE = os.path.join(a14.OUT, "hints")


def copies(mid, i, prepared, dmodel):
    """Per test copy: list of per-period dicts (cached, our own local files)."""
    path = os.path.join(CACHE, f"{mid}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    clf = joblib.load(os.path.join(a14.OUT, f"learned_fold_{mid}.joblib"))          # our own fold model
    fold = joblib.load(os.path.join(cv.OUT, f"fold_{a14.D_MODEL}_{mid}.joblib"))    # our own fold model
    m = load_match(mid)
    out = []
    for j, c in enumerate(cv.TEST_COINCIDENCE):
        per = []
        for q, f in zip(a14.test_periods(m, i, j, c), prepared[mid]["test"][c].feats):
            qs, F = a14.feats(q, m)
            per.append(dict(p=learned.p_home(clf, F), gt=truth_codes(q), t=q.t,
                            dead=decode_dead(dmodel.predict(fold["fitted"], f), m.fps, **fold["params"]),
                            true_dead=q.dead.copy(), home_owns=q.home_owns.copy()))
        out.append(per)
    os.makedirs(CACHE, exist_ok=True)
    joblib.dump(out, path)
    return out


def restarts(x, ev, period, fps):
    """[(frame, home, type)] for every true dead -> alive transition."""
    d = x["true_dead"]
    f = np.flatnonzero(d[:-1] & ~d[1:]) + 1
    sp = ev[(ev.period == period) & ev.set_piece.notna()]
    out = []
    for k in f:
        dt = np.abs(sp.t.to_numpy() - x["t"][k])
        typ = sp.set_piece.iloc[int(np.argmin(dt))] if len(dt) and dt.min() <= TYPE_MATCH_S else None
        out.append((int(k), bool(x["home_owns"][k]), typ))
    return out


def main():
    warnings.simplefilter("ignore")
    ids = list(cv.default_ids())
    prepared = None
    dmodel = cv.Model(drop=cv.DEFAULT_DROP)
    rows, types, first = [], {}, []
    for i, mid in enumerate(ids):
        if not os.path.exists(os.path.join(CACHE, f"{mid}.joblib")) and prepared is None:
            prepared = cv.prepare(ids, 2, cv.DEFAULT_MOTION, cv.DEFAULT_BLACKOUT)
        cps = copies(mid, i, prepared, dmodel)
        ev = a04.dfl_events(mid)
        fps = 25.0
        rng = np.random.default_rng([cv.SEED, i, 900])
        for j, per in enumerate(cps):
            rs = [restarts(x, ev, k, fps) for k, x in enumerate(per, start=1)]
            if j == 0:
                for r in rs:
                    for _, _, typ in r:
                        types[typ] = types.get(typ, 0) + 1
            gts = [x["gt"] for x in per]
            base = [learned.decode(x["p"], x["dead"], learned.SWITCH_COST) for x in per]
            f1, cr = a14.spell_f1(gts, base)
            rows.append(dict(match=mid, copy=j, hints="none", flip=0.0, f1=f1, cr=cr, n_stretch=0, n_hinted=0))
            for name, allowed in SETS.items():
                for flip in FLIPS:
                    preds, n_st, n_h = [], 0, 0
                    for x, r, b in zip(per, rs, base):
                        h = int(TAG_HALF_S * fps)
                        hint = [(k - h, k + h, home != (rng.random() < flip)) for k, home, typ in r if typ in allowed]
                        starts = learned.restart_starts(x["dead"], hint, int(HINT_WINDOW_S * fps))
                        n_st += len(learned.live_stretches(x["dead"]))
                        n_h += len(starts)
                        if flip == 0.0 and name == list(SETS)[-1]:
                            for s, home in starts.items():
                                first.append(dict(model_right=(b[s] == (learned.H if home else learned.A))))
                        preds.append(learned.decode(x["p"], x["dead"], learned.SWITCH_COST, starts))
                    f1, cr = a14.spell_f1(gts, preds)
                    rows.append(dict(match=mid, copy=j, hints=name, flip=flip, f1=f1, cr=cr,
                                     n_stretch=n_st, n_hinted=n_h))
        print(f"  done {mid}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a14.OUT, "restart_hints.csv"), index=False)     # per-match aggregates, local
    base = df[df.hints == "none"].groupby("match").f1.mean()
    print(f"\n{len(ids)} held-out matches x {len(cv.TEST_COINCIDENCE)} copies; switch cost {learned.SWITCH_COST:g}.")
    print("true restarts by DFL set-piece type (copy 0): "
          + ", ".join(f"{k}: {v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1])))
    fr = pd.DataFrame(first)
    print(f"hinted stretches (all set pieces, no flips) where the model already starts with the right team: "
          f"{fr.model_right.mean():.1%} of {len(fr)}")
    print(f"\nbaseline (no hints): spellF1 {base.mean():.3f}, count ratio {df[df.hints == 'none'].cr.mean():.2f}")
    print(f"{'hints':<30} {'flip':>5} {'hinted':>7} {'spellF1':>8} {'delta':>7} {'better':>7} {'count r':>8}")
    for (name, flip), g in df[df.hints != "none"].groupby(["hints", "flip"], sort=False):
        per = g.groupby("match").f1.mean()
        d = per - base
        print(f"{name:<30} {flip:>5.2f} {g.n_hinted.sum() / g.n_stretch.sum():>7.1%} {per.mean():>8.3f} "
              f"{d.mean():>+7.3f} {int((d > 0).sum()):>4}/{len(d)} {g.cr.mean():>8.2f}")


if __name__ == "__main__":
    main()
