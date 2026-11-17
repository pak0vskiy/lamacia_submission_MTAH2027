"""Dead score → dead mask → H/A/D codes → official metrics."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

H, A, D, X = (pm.LABELS.index(k) for k in "HADX")


def _runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))   # half-open


def decode_dead(prob, fps, enter, exit, min_dead_s=2.0, min_alive_s=1.0):
    """Hysteresis, then merge dead runs split by < min_alive_s, then drop dead runs < min_dead_s."""
    state = np.full(len(prob), np.nan)
    state[prob >= enter] = 1.0
    state[prob < exit] = 0.0
    if len(state) and np.isnan(state[0]):
        state[0] = 0.0
    idx = np.where(np.isnan(state), 0, np.arange(len(state)))
    dead = state[np.maximum.accumulate(idx)] == 1.0
    min_alive = int(round(min_alive_s * fps))
    for s, e in _runs(~dead):
        if 0 < s and e < len(dead) and e - s < min_alive:
            dead[s:e] = True
    min_dead = int(round(min_dead_s * fps))
    for s, e in _runs(dead):
        if e - s < min_dead:
            dead[s:e] = False
    return dead


def truth_codes(p):
    return np.where(p.dead, D, np.where(p.home_owns, H, A)).astype(np.int8)


def oracle_codes(p, pred_dead):
    """Setting A: predicted D → D; else DFL owner, carried from the nearest alive truth frame."""
    owner = np.where(p.home_owns, H, A).astype(np.int8)
    alive = ~p.dead
    n = len(owner)
    if alive.any():
        last = np.maximum.accumulate(np.where(alive, np.arange(n), -1))
        nxt = np.minimum.accumulate(np.where(alive, np.arange(n), n)[::-1])[::-1]
        src = np.where(last >= 0, last, nxt)
        owner = owner[src]
    return np.where(pred_dead, D, owner).astype(np.int8)


def concat(codes_per_period):
    sep = np.array([X], np.int8)
    out = []
    for i, c in enumerate(codes_per_period):
        if i:
            out.append(sep)
        out.append(np.asarray(c, np.int8))
    return np.concatenate(out)


def _d_spell_pr(gt, pred, runs, iou=0.5):
    def d_spells(codes):
        return [s for s in pm.segments(codes, runs) if s[2] == D]
    g, p = d_spells(gt), d_spells(pred)
    tp, fp, fn = pm.match_spells(pm.spell_overlaps(g, p), len(g), len(p), iou)
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def score(gt, pred, bootstrap_n=0):
    r = pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)
    runs = pm.valid_runs(gt)
    prec, rec = _d_spell_pr(gt, pm.fill_abstentions(pred, runs), runs)
    keys = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells",
            "spell_bootstrap_low", "spell_bootstrap_high")
    out = {k: r[k] for k in keys}
    out.update(d_spell_precision=prec, d_spell_recall=rec)
    return out
