"""Does the possession convention we tune on (DFL) match Metrica's? Aggregates only.

    uv run python analysis/16_metrica_convention.py

Other games only: IDSSE / DFL (7 matches) and Metrica sample games 1-3. Nothing from BAR-ATH.

Metrica's open data has no owner flag, so both providers are compared through the same lens, an
event recipe (dead_ball/metrica.py): D = [stoppage start, set piece); owner = team of the last
on-ball event, forward-filled (duels and loose balls keep the last team).

1. Label statistics: possessions per live minute, length quantiles, share of short possessions,
   for the DFL flag, the DFL event recipe and the Metrica event recipe. On DFL, how well the
   recipe reproduces the flag (spellF1, recipe as prediction).
2. The learned team model (trained on IDSSE only) scored by switch cost, each label set decoded
   inside its own dead mask (team logic only):
   - IDSSE held-out (fold models, c = 0.5 test copy): against the DFL flag and the DFL recipe;
   - Metrica (final model on all 7 IDSSE matches): against the Metrica recipe, ball as given and
     ATD-like (calibrated blackouts + 0.25 s smoothing).
   The question is whether the best cost moves between "DFL recipe" and "Metrica recipe": both
   use the same recipe, so a shift would be a provider difference, not a recipe artefact.

Decision rule, fixed before running: keep SWITCH_COST unless, on Metrica, cost 6 scores more than
0.02 below the best cost AND the Metrica best differs from the DFL-recipe best.
"""

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
from dead_ball.degrade import calibrated_mask, degrade, smooth_ball  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.metrica import GAMES, _conv, load_game  # noqa: E402
from dead_ball.pipeline import A, H, X, truth_codes  # noqa: E402
from possession import learned  # noqa: E402
from possession.vidal_codina import TUNED, period_codes  # noqa: E402

COSTS = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
OUT = os.path.join(cv.OUT, "..", "possession")
DFL_OWNER = {"PASS", "SHOT", "RECOVERY"}
DFL_OWNER_GENERIC = {"OtherBallAction"}
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def spell_f1(gts, preds):
    sep = np.array([X], np.int8)
    g = np.concatenate([np.r_[sep, x] if i else x for i, x in enumerate(gts)])
    p = np.concatenate([np.r_[sep, x] if i else x for i, x in enumerate(preds)])
    runs = pm.valid_runs(g)
    r = pm.spell_scores(g, pm.fill_abstentions(p, runs), runs)
    return r["mean_f1"], r["n_pred"] / r["n_gt"]


def possession_stats(code_list, fps):
    lens, live = [], 0
    for c in code_list:
        live += int(np.isin(c, (H, A)).sum())
        e = np.flatnonzero(np.diff(np.r_[-1, c, -1]))
        lens += [(t - s) / fps for s, t in zip(e[:-1], e[1:]) if c[s] in (H, A)]
    lens = np.array(lens)
    return {"possessions": len(lens), "live min": live / fps / 60,
            "per live min": len(lens) / (live / fps / 60),
            "< 2 s": np.mean(lens < 2), "< 5 s": np.mean(lens < 5),
            "len q10/25/50/75/90 (s)": "/".join(f"{x:.1f}" for x in np.quantile(lens, QS))}


# ---------------------------------------------------------------- DFL event recipe

def dfl_recipe(mid, m):
    """Per period: (dead, home_owns) from DFL events, on the tracking clock."""
    from kloppy import sportec
    conv = _conv()
    _, meta, evf = conv.dfl_files(mid)
    ds = sportec.load_event(event_data=evf, meta_data=meta)
    home = ds.metadata.teams[0]
    own = [(e.period.id, e.timestamp.total_seconds(), e.team == home) for e in ds.events
           if e.team is not None and (e.event_type.name in DFL_OWNER
                                      or getattr(e, "event_name", None) in DFL_OWNER_GENERIC)]
    own = pd.DataFrame(own, columns=["period", "t", "home"])
    ev = conv.dfl_events(mid)
    spells = conv.recipe_spells(ev)
    out = []
    for k, p in enumerate(m.periods, start=1):
        g, o = ev[ev.period == k], own[own.period == k].sort_values("t", kind="stable")
        dead = (p.t < g.t.min()) | (p.t > g.t.max())
        for per, s, e, _ in spells:
            if per == k:
                dead |= (p.t >= s) & (p.t < e)
        idx = np.clip(np.searchsorted(o.t.to_numpy(), p.t, side="right") - 1, 0, None)
        out.append((dead, o.home.to_numpy()[idx]))
    return out


def sweep(per, labelsets):
    """per: list of dicts with p, q; labelsets: name -> list of (dead, home_owns) per period."""
    rows = []
    for name, labs in labelsets.items():
        gts = [truth_codes(replace(x["q"], dead=d, home_owns=h)) for x, (d, h) in zip(per, labs)]
        for cost in COSTS:
            f1, cr = spell_f1(gts, [learned.decode(x["p"], d, cost) for x, (d, _) in zip(per, labs)])
            rows.append(dict(labels=name, cost=cost, f1=f1, count_ratio=cr))
        f1, cr = spell_f1(gts, [period_codes(x["q"], d, x["fps"], TUNED) for x, (d, _) in zip(per, labs)])
        rows.append(dict(labels=name, cost="VC", f1=f1, count_ratio=cr))
    return rows


def main():
    warnings.simplefilter("ignore")
    ids = list(cv.default_ids())
    final = os.path.join(OUT, "learned_final.joblib")   # written by analysis/15 (our own local file)
    if not os.path.exists(final):
        raise SystemExit("run analysis/15_learned_on_atd.py first (it fits learned_final.joblib)")
    clf = joblib.load(final)

    # ---- IDSSE: flag vs event recipe, labels and model
    flag_codes, recipe_codes, rows, recipe_vs_flag = [], [], [], []
    j = cv.TEST_COINCIDENCE.index(0.5)
    for i, mid in enumerate(ids):
        m = load_match(mid)
        rec = dfl_recipe(mid, m)
        fc = [truth_codes(p) for p in m.periods]
        rc = [truth_codes(replace(p, dead=d, home_owns=h)) for p, (d, h) in zip(m.periods, rec)]
        flag_codes += fc
        recipe_codes += rc
        recipe_vs_flag.append(spell_f1(fc, rc))
        fclf = joblib.load(os.path.join(OUT, f"learned_fold_{mid}.joblib"))
        rng = np.random.default_rng([cv.SEED, i, 100 + j])
        per = []
        for p in m.periods:
            q = degrade(p, m.fps, rng, 0.5, motion=cv.DEFAULT_MOTION, blackout=cv.DEFAULT_BLACKOUT)
            q = replace(q, ball=smooth_ball(q.ball, m.fps, learned.BALL_SMOOTH_S))
            F = learned.team_features(q.ball, q.home, q.away, m.fps, m.half_len, m.half_wid)
            per.append(dict(q=q, p=learned.p_home(fclf, F), fps=m.fps))
        for r in sweep(per, {"DFL flag": [(p.dead, p.home_owns) for p in m.periods], "DFL recipe": rec}):
            rows.append(dict(data="IDSSE held-out", match=mid, **r))
        print(f"  done {mid}", flush=True)
        del m

    # ---- Metrica: recipe labels and model
    met_codes, acc = [], []
    for g in GAMES:
        m = load_game(g)
        met_codes += [truth_codes(p) for p in m.periods]
        for variant in ("ball as given", "ATD-like"):
            rng = np.random.default_rng([cv.SEED, 1000 + g])
            per = []
            for p in m.periods:
                q = p
                if variant == "ATD-like":
                    ball = p.ball.copy()
                    ball[calibrated_mask(p, m.fps, rng)] = np.nan
                    q = replace(p, ball=smooth_ball(ball, m.fps, learned.BALL_SMOOTH_S))
                F = learned.team_features(q.ball, q.home, q.away, m.fps, m.half_len, m.half_wid)
                pr = learned.p_home(clf, F)
                per.append(dict(q=q, p=pr, fps=m.fps))
                live, vis = ~p.dead, F.ball_visible.to_numpy() > 0
                ok = (pr >= 0.5) == p.home_owns
                for part, mask in (("visible", live & vis), ("hidden", live & ~vis)):
                    acc.append(dict(variant=variant, part=part, n=int(mask.sum()), correct=int(ok[mask].sum())))
            for r in sweep(per, {"Metrica recipe": [(p.dead, p.home_owns) for p in m.periods]}):
                rows.append(dict(data=f"Metrica, {variant}", match=g, **r))
        print(f"  done metrica {g}", flush=True)
        del m

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrica_convention.csv"), index=False)   # per-match aggregates, local

    print("\n1. Label statistics (live = not D under each label set's own dead mask)")
    stats = pd.DataFrame({"DFL flag (7)": possession_stats(flag_codes, 25.0),
                          "DFL event recipe (7)": possession_stats(recipe_codes, 25.0),
                          "Metrica event recipe (3)": possession_stats(met_codes, 25.0)})
    print(stats.to_string())
    f1s = np.array(recipe_vs_flag)
    print(f"\nDFL recipe scored against the DFL flag: spellF1 {f1s[:, 0].mean():.3f} "
          f"(min {f1s[:, 0].min():.3f}), count ratio {f1s[:, 1].mean():.2f}")

    print("\n2. Learned model by switch cost (mean over matches; decoded inside each label set's D)")
    pd.set_option("display.width", 220)
    for (data, labs), d in df.groupby(["data", "labels"], sort=False):
        mean = d.groupby("cost", sort=False)[["f1", "count_ratio"]].mean()
        num = mean.drop(index="VC")
        best = num.f1.astype(float).idxmax()
        print(f"\n{data} vs {labs}: best cost {best:g} ({num.f1.max():.3f}); cost 6: {num.f1.loc[6.0]:.3f}; "
              f"VC: {mean.f1.loc['VC']:.3f} (count ratio {mean.count_ratio.loc['VC']:.2f})")
        print("  f1:          " + " ".join(f"{k:g}:{v:.3f}" for k, v in num.f1.items()))
        print("  count ratio: " + " ".join(f"{k:g}:{v:.2f}" for k, v in num.count_ratio.items()))
        per_match = d[d.cost != "VC"].pivot_table(index="match", columns="cost", values="f1")
        print("  best cost per match: " + ", ".join(f"{k}:{v:g}" for k, v in per_match.idxmax(axis=1).items()))

    a = pd.DataFrame(acc).groupby(["variant", "part"])[["n", "correct"]].sum()
    print("\nMetrica, per-frame H/A accuracy of the learned model (argmax) on recipe-live frames:")
    print((a.correct / a.n).round(3).to_string())


if __name__ == "__main__":
    main()
