"""Leave-one-match-out CV on degraded IDSSE. Per-frame data stays in memory / artifacts/ only."""

import itertools
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from dead_ball.degrade import degrade
from dead_ball.features import frame_features
from dead_ball.idsse import CACHE, MATCH_IDS, load_match
from dead_ball.pipeline import concat, decode_dead, oracle_codes, score, truth_codes
from dead_ball.v1_adapter import v1_codes_from_frame, v1_frame

SEED = 0
# Chosen configuration (observations/2026-11-04_dead-ball-transfer.md): no ball-speed feature,
# 'mixed' dead-ball motion in the degradation.
DEFAULT_DROP = ("ball_spd_1",)
DEFAULT_MOTION = "mixed"
# Ball blackouts fitted to BAR-ATH tracking (observations/2026-11-05_blackout-calibration.md).
DEFAULT_BLACKOUT = "calibrated"
TEST_COINCIDENCE = (0.0, 0.5, 0.9)
TRAIN_COINCIDENCE = (0.0, 0.9)
SUBSAMPLE = 5
OUT = os.path.join(CACHE, "dead_ball")   # joblib files here are written by this module only (trusted, local)
GRID = dict(enter=(0.5, 0.6, 0.7), exit=(0.3, 0.4, 0.5), min_dead_s=(1.0, 2.0, 3.0))


@dataclass
class Copy:
    """One degraded version of a match: per-period features, labels, and what scoring needs."""
    match_id: str
    coincidence: float
    feats: list      # per period DataFrame
    periods: list    # per period Period, labels only (positions dropped)
    v1: list         # per period v1 feature frame


def _strip(p):
    from dataclasses import replace
    e = np.empty((len(p), 0, 2), np.float32)
    return replace(p, ball=p.ball[:, :0], home=e, away=e)


def make_copy(m, rng, coincidence, motion="carry", blackout="coincidence"):
    feats, periods, v1 = [], [], []
    for p in m.periods:
        q = degrade(p, m.fps, rng, coincidence, motion=motion, blackout=blackout)
        feats.append(frame_features(q, m.fps, m.half_len, m.half_wid))
        v1.append(v1_frame(q, m.half_len, m.half_wid, m.fps))
        periods.append(_strip(q))
    return Copy(m.id, coincidence, feats, periods, v1)


def prepare(match_ids, n_copies, motion="carry", blackout="coincidence"):
    """{match: {'train': [Copy], 'test': {c: Copy}, 'match': Match-without-positions}}. Cached.
    blackout='calibrated': BAR-ATH-fitted blackouts; the coincidence values are then just labels of
    three independent test copies."""
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for i, mid in enumerate(match_ids):
        tag = "v2" if motion == "carry" else f"v3{motion}"
        tag += "cal" if blackout == "calibrated" else ""
        path = os.path.join(OUT, f"prepared_{tag}_{mid}_{n_copies}.joblib")
        if not os.path.exists(path):
            m = load_match(mid)
            train = []
            for k in range(n_copies):
                rng = np.random.default_rng([SEED, i, k])
                train.append(make_copy(m, rng, rng.uniform(*TRAIN_COINCIDENCE), motion, blackout))
            test = {c: make_copy(m, np.random.default_rng([SEED, i, 100 + j]), c, motion, blackout)
                    for j, c in enumerate(TEST_COINCIDENCE)}
            m.periods = []
            joblib.dump(dict(train=train, test=test, match=m), path)
        data[mid] = joblib.load(path)
    return data


def stack(copies, subsample=SUBSAMPLE):
    X = pd.concat([f.iloc[::subsample] for c in copies for f in c.feats], ignore_index=True)
    y = np.concatenate([p.dead[::subsample] for c in copies for p in c.periods])
    return X, y


# ---------------------------------------------------------------- methods

class Model:
    """Gradient-boosted trees; `drop` removes feature columns (ablations)."""

    def __init__(self, drop=(), name=None):
        self.drop = list(drop)
        self.name = name or ("model" if not drop else "model-no-" + "-".join(drop))

    def _x(self, X):
        return X.drop(columns=self.drop)

    def fit(self, X, y):
        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=31, early_stopping=False)
        return clf.fit(self._x(X), y)

    def predict(self, fitted, X):
        return fitted.predict_proba(self._x(X))[:, 1]


class NoDead:
    """Floor: never predicts D."""
    name = "none"

    def fit(self, X, y):
        return None

    def predict(self, fitted, X):
        return np.zeros(len(X))


# ---------------------------------------------------------------- scoring

def copy_codes(copy, match, probs, params, setting):
    """Predicted and true codes for one copy. setting 'A' (oracle teams), 'B' (v1), 'B0' (v1 as is)."""
    gt, pred = [], []
    probs = probs if probs is not None else [None] * len(copy.periods)
    for p, f1, pr in zip(copy.periods, copy.v1, probs):
        gt.append(truth_codes(p))
        if setting == "B0":
            pred.append(v1_codes_from_frame(f1, match, None))
            continue
        dead = decode_dead(pr, match.fps, **params)
        pred.append(oracle_codes(p, dead) if setting == "A" else
                    v1_codes_from_frame(f1, match, dead.astype(float)))
    return concat(gt), concat(pred)


def tune_decoder(method, data, train_ids):
    """Decoder params by mean setting-A spellF1 on inner out-of-fold predictions (3 folds by match)."""
    folds = [f for f in np.array_split(np.array(train_ids), 3) if len(f)]
    preds = []
    for held in folds:
        inner = [m for m in train_ids if m not in held]
        fitted = method.fit(*stack([c for m in inner for c in data[m]["train"]]))
        for m in held:
            c = data[m]["train"][0]
            preds.append((c, data[m]["match"], [method.predict(fitted, f) for f in c.feats]))
    best, best_score = None, -1
    for enter, exit, md in itertools.product(*GRID.values()):
        if exit > enter:
            continue
        params = dict(enter=enter, exit=exit, min_dead_s=md)
        s = np.mean([score(*copy_codes(c, m, pr, params, "A"))["spell_f1_mean"] for c, m, pr in preds])
        if s > best_score:
            best, best_score = params, s
    return best


def run_cv(method, data, bootstrap_n=0):
    """Rows: (method, held-out match, coincidence, setting, metrics...). Saves fold models + probs."""
    ids = list(data)
    rows = []
    for held in ids:
        train_ids = [m for m in ids if m != held]
        params = (tune_decoder(method, data, train_ids) if method.name != "none"
                  else dict(enter=0.5, exit=0.5, min_dead_s=2.0))
        fitted = method.fit(*stack([c for m in train_ids for c in data[m]["train"]]))
        joblib.dump(dict(fitted=fitted, params=params),
                    os.path.join(OUT, f"fold_{method.name}_{held}.joblib"))
        match = data[held]["match"]
        for c, copy in data[held]["test"].items():
            probs = [method.predict(fitted, f) for f in copy.feats]
            if c == 0.5:
                joblib.dump(probs, os.path.join(OUT, f"oof_{method.name}_{held}.joblib"))
            for setting in ("A", "B"):
                r = score(*copy_codes(copy, match, probs, params, setting), bootstrap_n=bootstrap_n)
                rows.append(dict(method=method.name, match=held, coincidence=c, setting=setting,
                                 **params, **r))
    return pd.DataFrame(rows)


def run_v1_reference(data):
    rows = []
    for held, d in data.items():
        for c, copy in d["test"].items():
            r = score(*copy_codes(copy, d["match"], None, None, "B0"))
            rows.append(dict(method="v1_as_is", match=held, coincidence=c, setting="B", **r))
    return pd.DataFrame(rows)


def default_ids(n=None):
    return MATCH_IDS[:n] if n else MATCH_IDS
