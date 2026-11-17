"""Leave-one-match-out CV of the dead-ball classifier on degraded IDSSE. Aggregates only.

    uv run python analysis/06_dead_ball_cv.py [--matches N] [--copies K] [--method model|rules]

Settings: A = DFL-oracle teams (isolates D), B = Sam's v1 team logic with our D; v1_as_is = v1
unmodified (reference); none = never D (floor). Other games only; nothing from BAR-ATH is read.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dead_ball import cv  # noqa: E402

COLS = ["spell_f1_mean", "macro_f1", "d_spell_precision", "d_spell_recall"]


def methods(name, drop):
    if name == "rules":
        from dead_ball.rules import Rules
        return [Rules()]
    return [cv.Model(drop=drop), cv.NoDead()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=None)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--method", default="model")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--drop", nargs="*", default=list(cv.DEFAULT_DROP),
                    help="feature columns to remove (default: the chosen config; pass --drop alone for none)")
    ap.add_argument("--tag", default="", help="suffix for the output CSV")
    ap.add_argument("--motion", default=cv.DEFAULT_MOTION, choices=["carry", "mixed"],
                    help="how the DFL-frozen dead ball is replaced in the degradation")
    ap.add_argument("--blackout", default=cv.DEFAULT_BLACKOUT, choices=["coincidence", "calibrated"],
                    help="ball blackout model in the degradation (calibrated = fitted to BAR-ATH, analysis/12)")
    a = ap.parse_args()

    data = cv.prepare(cv.default_ids(a.matches), a.copies, a.motion, a.blackout)
    ms = methods(a.method, a.drop)
    if a.motion != "carry":
        for m in ms:
            m.name = f"{m.name}@{a.motion}"
    if a.blackout == "calibrated":
        for m in ms:
            m.name = f"{m.name}+cal"
    frames = [cv.run_cv(m, data, a.bootstrap) for m in ms]
    if a.method == "model":
        frames.append(cv.run_v1_reference(data))
    df = pd.concat(frames, ignore_index=True)
    df["count_ratio"] = df.n_pred_spells / df.n_gt_spells
    os.makedirs(cv.OUT, exist_ok=True)
    df.to_csv(os.path.join(cv.OUT, f"cv_{ms[0].name}{a.tag}.csv"), index=False)   # per-match aggregates, local

    pd.set_option("display.width", 200)
    g = df.groupby(["setting", "method", "coincidence"])[COLS + ["count_ratio"]]
    print(f"{len(data)} held-out matches, {a.copies} training copies per match\n")
    print("mean over held-out matches")
    print(g.mean().round(3).to_string())
    print("\nstd over held-out matches")
    print(g.std().round(3).to_string())
    if "enter" in df:
        print("\ndecoder params chosen per fold (inner CV):")
        print(df.dropna(subset=["enter"]).drop_duplicates(["method", "match"])
              .groupby(["method", "enter", "exit", "min_dead_s"]).size().to_string())


if __name__ == "__main__":
    main()
