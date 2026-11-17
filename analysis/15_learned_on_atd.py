"""Does the learned team logic behave on our ATD like on held-out IDSSE? Aggregates only, no labels.

    uv run python analysis/15_learned_on_atd.py

BAR-ATH side: the submission's D model (scripts/make_submission.py cache) and an owner model trained
on all 7 IDSSE matches' training copies (analysis/14 cache), switch cost COST. No GT, public or
private. IDSSE side: each match's c = 0.5 test copy with its own held-out fold models from
analysis/14 and the D fold models. Compared: the model's confidence, feature ranges, possession count
and length, home share, agreement with VC. Plausibility only. Nothing is tuned on BAR-ATH.
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analysis.atd_cache import graded_periods  # noqa: E402
from dead_ball import cv  # noqa: E402
from dead_ball.degrade import degrade, smooth_ball  # noqa: E402
from dead_ball.features import frame_features  # noqa: E402
from dead_ball.idsse import load_match  # noqa: E402
from dead_ball.pipeline import A, D, H, decode_dead  # noqa: E402
from possession import learned  # noqa: E402
from possession.vidal_codina import TUNED, period_codes  # noqa: E402
import make_submission as ms  # noqa: E402

COST = learned.SWITCH_COST
FPS = 25.0
OUT = os.path.join(cv.OUT, "..", "possession")
D_MODEL = "model-no-ball_spd_1@mixed+cal"
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
FEATS = ("home_d1", "home_since_gain", "home_zone_past1", "home_adv_v1", "home_speed1",
         "home_spread", "since_ball", "ball_speed")


def stats(per, fps):
    """per: list of (p, dead, learned codes, vc codes, F)."""
    live_p = np.concatenate([p[~d] for p, d, *_ in per])
    lens, n_live = [], 0
    home = away = agree = 0
    for p, d, c, v, _ in per:
        n_live += int((~d).sum())
        e = np.flatnonzero(np.diff(np.r_[-1, c, -1]))
        for s, t in zip(e[:-1], e[1:]):
            if c[s] in (H, A):
                lens.append((t - s) / fps)
        home += int((c == H).sum())
        away += int((c == A).sum())
        vv = v[~d]
        known = np.isin(vv, (H, A))
        agree += int((vv[known] == c[~d][known]).sum())
        n_known = int(known.sum())
    lens = np.array(lens)
    r = {"live min": n_live / fps / 60,
         "poss / live min": len(lens) / (n_live / fps / 60),
         "home share": home / max(home + away, 1),
         "p in [0.3,0.7]": np.mean((live_p > 0.3) & (live_p < 0.7)),
         "p in [0.1,0.9]": np.mean((live_p > 0.1) & (live_p < 0.9)),
         "agree with VC (VC known)": agree / max(sum(int(np.isin(v[~d], (H, A)).sum()) for _, d, _, v, _ in per), 1)}
    for q, x in zip(QS, np.quantile(lens, QS)):
        r[f"poss len q{int(q * 100)} (s)"] = x
    F = pd.concat([f[~d] for _, d, _, _, f in per], ignore_index=True)
    for k in FEATS:
        qq = np.nanquantile(F[k].to_numpy(), (0.1, 0.5, 0.9))
        r[f"{k} q10/50/90"] = "/".join(f"{x:.2f}" for x in qq)
    return r


def main():
    warnings.simplefilter("ignore")
    ids = list(cv.default_ids())
    # final owner model on all 7 (rows cached by analysis/14; our own local files)
    final = os.path.join(OUT, "learned_final.joblib")
    if not os.path.exists(final):
        rows = [joblib.load(os.path.join(OUT, f"learned_train_{m}.joblib")) for m in ids]
        joblib.dump(learned.fit(pd.concat([r[0] for r in rows], ignore_index=True),
                                np.concatenate([r[1] for r in rows])), final)
    clf = joblib.load(final)

    # BAR-ATH
    bundle = ms.train_dead_model()
    method = cv.Model(drop=bundle["drop"])
    periods, hl, hw = graded_periods()
    bar = []
    for p in periods:
        dead = decode_dead(method.predict(bundle["fitted"], frame_features(p, FPS, hl, hw)), FPS, **bundle["params"])
        F = learned.team_features(p.ball, p.home, p.away, FPS, hl, hw)
        pr = learned.p_home(clf, F)
        bar.append((pr, dead, learned.decode(pr, dead, COST), period_codes(p, dead, FPS, TUNED), F))

    # IDSSE, c = 0.5 test copy, held-out fold models
    prepared = cv.prepare(ids, 2, cv.DEFAULT_MOTION, cv.DEFAULT_BLACKOUT)
    j = cv.TEST_COINCIDENCE.index(0.5)
    ids_per = []
    dm = cv.Model(drop=cv.DEFAULT_DROP)
    for i, mid in enumerate(ids):
        fclf = joblib.load(os.path.join(OUT, f"learned_fold_{mid}.joblib"))
        fold = joblib.load(os.path.join(cv.OUT, f"fold_{D_MODEL}_{mid}.joblib"))
        m = load_match(mid)
        rng = np.random.default_rng([cv.SEED, i, 100 + j])
        for p, f in zip(m.periods, prepared[mid]["test"][0.5].feats):
            q = degrade(p, m.fps, rng, 0.5, motion=cv.DEFAULT_MOTION, blackout=cv.DEFAULT_BLACKOUT)
            q = replace(q, ball=smooth_ball(q.ball, m.fps, 0.25))
            dead = decode_dead(dm.predict(fold["fitted"], f), m.fps, **fold["params"])
            F = learned.team_features(q.ball, q.home, q.away, m.fps, m.half_len, m.half_wid)
            pr = learned.p_home(fclf, F)
            ids_per.append((pr, dead, learned.decode(pr, dead, COST), period_codes(q, dead, m.fps, TUNED), F))
        del m

    a, b = stats(ids_per, FPS), stats(bar, FPS)
    print(f"\nlearned team logic, switch cost {COST}; live = our D model's calls on both sides\n")
    print(f"{'':34s} {'IDSSE held-out (7)':>22s} {'BAR-ATH':>22s}")
    for k in a:
        fa, fb = a[k], b[k]
        fmt = (lambda x: f"{x:22.3f}") if isinstance(fa, float) else (lambda x: f"{x:>22s}")
        print(f"{k:34s} {fmt(fa)} {fmt(fb)}")
    print(f"\n(IDSSE live min is the total over 7 matches; BAR-ATH is one match.)")


if __name__ == "__main__":
    main()
