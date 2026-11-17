"""Regenerate the submission from scratch: one command, deterministic.

    uv run python scripts/make_submission.py [--out submissions/submission.csv] [--retrain]

Pipeline (current choice, see docs/decisions.md):
1. Dead-ball model: gradient-boosted trees on tracking-only features, trained on all 7 IDSSE
   matches (other games), degraded like our ATD. Decoder thresholds picked by inner 3-fold CV by
   match, the same procedure as each fold of analysis/06. Config = dead_ball.cv defaults.
2. Team logic (default `--team learned`): per-frame P(home has the ball) from gradient-boosted
   trees on positions-only features (possession/learned.py), trained on live frames of all 7
   IDSSE matches degraded and smoothed like our ATD, then a 2-state Viterbi with switch cost
   learned.SWITCH_COST inside each predicted live stretch. `--team vc` gives the previous
   Vidal-Codina rule (possession.vidal_codina.TUNED).
3. Opening kickoff (default on, `--no-kickoff` off): positions-only kickoff detector
   (dead_ball/kickoff.py, validated on other games) gated by Smart Tagging's automatic
   `Kick Off (Start)` intervals; everything in a graded window before the opening kick is D.
4. BAR-ATH graded windows → H/A/D; everything else X.

Nothing here reads any ground truth, public or private. Output is aggregates only on stdout; the
frame-level CSV goes to submissions/ (gitignored). The trained model is cached in
artifacts/submission/ under a key of its configuration; --retrain ignores the cache.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.atd_cache import GRADED, graded_periods  # noqa: E402
from dead_ball import cv, kickoff  # noqa: E402
from dead_ball.features import frame_features  # noqa: E402
from dead_ball.pipeline import A, D, H, X, decode_dead  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from possession import learned  # noqa: E402
from possession.vidal_codina import TUNED, period_codes  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as smart_tagging  # noqa: E402

# Params
FPS = 25.0
N_COPIES = 2
OUT_DIR = os.path.join(ROOT, "artifacts", "submission")   # our own joblib files only (trusted, local)
DEFAULT_OUT = os.path.join(ROOT, "submissions", "submission.csv")
KICK_SEARCH_S = (120.0, 60.0)   # search this long before / after a tagged opening kickoff
KICK_GATE_S = 5.0               # accept a detected kick only this close to the tagged interval


def dead_model_config():
    return dict(ids=list(cv.default_ids()), copies=N_COPIES, motion=cv.DEFAULT_MOTION,
                blackout=cv.DEFAULT_BLACKOUT,
                drop=list(cv.DEFAULT_DROP), seed=cv.SEED, subsample=cv.SUBSAMPLE,
                grid={k: list(v) for k, v in cv.GRID.items()})


def train_dead_model(retrain=False):
    """Final D model on all IDSSE matches + decoder params. Cached by config hash."""
    cfg = dead_model_config()
    if len(cfg["ids"]) != 7:
        sys.exit(f"expected 7 IDSSE matches in data/external/idsse, found {len(cfg['ids'])} (docs/idsse-and-dead-ball-model.md)")
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    path = os.path.join(OUT_DIR, f"dead_model_{key}.joblib")
    if os.path.exists(path) and not retrain:
        print(f"dead-ball model: cached {os.path.relpath(path, ROOT)}")
        return joblib.load(path)
    print("dead-ball model: training on all 7 IDSSE matches (decoder by inner 3-fold CV) ...")
    data = cv.prepare(cfg["ids"], N_COPIES, cv.DEFAULT_MOTION, cv.DEFAULT_BLACKOUT)
    method = cv.Model(drop=cv.DEFAULT_DROP)
    params = cv.tune_decoder(method, data, cfg["ids"])
    fitted = method.fit(*cv.stack([c for m in cfg["ids"] for c in data[m]["train"]]))
    bundle = dict(fitted=fitted, params=params, drop=list(cv.DEFAULT_DROP), config=cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def team_model_config():
    return dict(ids=list(cv.default_ids()), copies=learned.N_COPIES, subsample=learned.SUBSAMPLE,
                ball_smooth=learned.BALL_SMOOTH_S, seed=learned.SEED, motion=cv.DEFAULT_MOTION,
                blackout=cv.DEFAULT_BLACKOUT, version="learned-v1")


def train_team_model(retrain=False):
    """Owner model on live frames of all 7 IDSSE matches. Cached by config hash."""
    from dead_ball.idsse import load_match
    import pandas as pd
    cfg = team_model_config()
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    path = os.path.join(OUT_DIR, f"team_model_{key}.joblib")
    if os.path.exists(path) and not retrain:
        print(f"team model: cached {os.path.relpath(path, ROOT)}")
        return joblib.load(path)
    print("team model: training on all 7 IDSSE matches ...")
    rows = []
    for i, mid in enumerate(cfg["ids"]):
        rows.append(learned.training_rows(load_match(mid), i))
    clf = learned.fit(pd.concat([r[0] for r in rows], ignore_index=True), np.concatenate([r[1] for r in rows]))
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(clf, path)
    return clf


def opening_kick(p, s, tags):
    """Window-relative frame of the half's opening kick, or None.

    The kickoff detector (dead_ball/kickoff.py, positions only, validated on other games) runs
    around Smart Tagging's automatic `Kick Off (Start)` intervals, and a detection counts only if
    it lands within KICK_GATE_S of one. No frame constants."""
    for t in tags:
        lo = max(0, t["start_frame"] - s - int(KICK_SEARCH_S[0] * FPS))
        hi = min(len(p), t["end_frame"] - s + int(KICK_SEARCH_S[1] * FPS))
        if hi <= lo:
            continue
        k = kickoff.detect(p.home, p.away, FPS, lo, hi)
        gate = int(KICK_GATE_S * FPS)
        if k is not None and t["start_frame"] - gate <= s + k <= t["end_frame"] + gate:
            return k
    return None


def kickoff_tags():
    return [t for t in smart_tagging.load()
            if (t["code"] or "").strip().lower() == "set pieces"
            and (t["tags"].get("Type") or "").strip().lower() == "kick off (start)"]


def predict(bundle, team_clf=None, use_kickoff=True):
    """Codes for the whole video: X outside the graded windows. team_clf None → Vidal-Codina.
    use_kickoff: everything in a graded window before its detected opening kick is D."""
    method = cv.Model(drop=bundle["drop"])
    periods, hl, hw = graded_periods()
    tags = kickoff_tags() if use_kickoff else []
    out = np.full(GAMES[GAME]["n_video_frames"], X, np.int8)
    for (s, e), p in zip(GRADED, periods):
        dead = decode_dead(method.predict(bundle["fitted"], frame_features(p, FPS, hl, hw)), FPS,
                           **bundle["params"])
        k = opening_kick(p, s, tags) if use_kickoff else None
        if k:
            print(f"  window {s}-{e}: opening kick detected {k / FPS:.1f} s in; "
                  f"{int((~dead[:k]).sum()) / FPS:.1f} s of predicted live play before it set to D")
            dead[:k] = True
        if team_clf is None:
            out[s:e] = period_codes(p, dead, FPS, TUNED)
        else:
            prob = learned.p_home(team_clf, learned.team_features(p.ball, p.home, p.away, FPS, hl, hw))
            out[s:e] = learned.decode(prob, dead, learned.SWITCH_COST)
    return out


def write_csv(codes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = np.array(pm.LABELS)[codes]
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))


def summary(codes):
    graded = np.zeros(len(codes), bool)
    for s, e in GRADED:
        graded[s:e] = True
    g = codes[graded]
    return ", ".join(f"{k} {np.mean(g == c):.1%}" for k, c in zip("HADX", (H, A, D, X)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--retrain", action="store_true", help="ignore the cached models")
    ap.add_argument("--team", default="learned", choices=["learned", "vc"], help="team logic")
    ap.add_argument("--no-kickoff", action="store_true", help="don't force D before the opening kick")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    bundle = train_dead_model(a.retrain)
    team_clf = train_team_model(a.retrain) if a.team == "learned" else None
    print(f"  decoder {bundle['params']}; team "
          + (f"learned, switch cost {learned.SWITCH_COST}" if team_clf is not None else f"VC {TUNED}"))
    codes = predict(bundle, team_clf, use_kickoff=not a.no_kickoff)
    write_csv(codes, a.out)

    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"\nwrote {os.path.relpath(a.out, ROOT)}  sha256 {digest[:16]}")
    print(f"graded frames: {summary(codes)}")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), a.out],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
