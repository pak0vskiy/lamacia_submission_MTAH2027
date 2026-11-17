"""Smart Tagging vs our BAR-ATH predictions and vs geometry. Label-free, aggregates only.

    uv run python analysis/19_smart_tagging_on_atd.py [--sub submissions/submission.csv]

No ground truth is read. Three checks:
1. Team tags against geometry: a corner is taken by the team attacking that end, a goal kick by
   the team defending it. The end comes from where the ball (else the players) is during the tag;
   the attack direction from the ATD metadata. An estimate of how often a set-piece team tag is
   right, without labels.
2. Set pieces against our dead-ball calls: does a predicted live stretch start inside the tag
   (+- HINT_WINDOW_S)? And when it does, does our predicted team at that start match the tag?
3. Phase tags against our team calls: per code, the share of our predicted live frames inside the
   tag that carry the tagged team. Descriptive only (a phase's team is the team in possession for
   attacking codes, presumably the defending team for DEFENDING codes).
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.atd_cache import GRADED, _strip, load  # noqa: E402
from games import asset_paths  # noqa: E402
from possession import learned  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as smart_tagging  # noqa: E402

FPS = 25.0
HINT_WINDOW_S = 2.0
DEFAULT_SUB = os.path.join(ROOT, "submissions", "submission.csv")


def team_names():
    """Team names in metadata order (index 0 = home = H)."""
    root = ET.parse(asset_paths()["atd_meta"]).getroot()
    out = []
    for t in root.iter():
        if _strip(t.tag) == "Team":
            name = t.attrib.get("name") or next((c.text for c in t if _strip(c.tag) == "Name"), None)
            out.append((name or "").strip())
    return out


def side_of(tag_team, names):
    """'H' / 'A' / None for a Smart Tagging Team value."""
    if not tag_team or tag_team.strip().upper() == "N/A":
        return None
    key = tag_team.strip().lower()
    hits = [i for i, n in enumerate(names) if key in n.lower() or n.lower() in key
            or key.split()[0][:5] == n.lower().split()[-1][:5] or key.split()[0][:5] == n.lower().split()[0][:5]]
    return "HA"[hits[0]] if len(hits) == 1 else None


def graded_half(frame):
    for k, (s, e) in enumerate(GRADED):
        if s <= frame < e:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=DEFAULT_SUB)
    a = ap.parse_args()
    labels = pd.read_csv(a.sub).label.to_numpy()          # our own predictions, not ground truth
    atd = load()
    hl = float(atd["pitch"][0]) / 2
    home_plus_1st = bool(atd["home_attacks_plus_x_first_half"])
    team, gk = atd["slot_team"], atd["slot_gk"].astype(bool)

    names = team_names()
    tags = smart_tagging.load()
    vals = sorted({t["tags"].get("Team") for t in tags} - {None})
    print(f"metadata teams (H, A): {names}; Smart Tagging Team -> side: "
          + ", ".join(f"{v} -> {side_of(v, names)}" for v in vals))

    dead = labels == "D"
    starts = np.flatnonzero(~dead[1:] & dead[:-1]) + 1            # first frame of each predicted live stretch
    starts = starts[np.isin(labels[starts], ["H", "A"])]

    # 1 + 2: set pieces
    geo, sp = [], []
    for t in tags:
        if (t["code"] or "").strip().lower() != "set pieces":
            continue
        s, e = t["start_frame"], t["end_frame"]
        k = graded_half(s)
        typ = (t["tags"].get("Type") or "").strip()
        side = side_of(t["tags"].get("Team"), names)
        if k is None:
            continue
        w = int(HINT_WINDOW_S * FPS)
        inside = starts[(starts >= s - w) & (starts <= e + w)]
        pred = labels[inside[0]] if len(inside) else None
        sp.append(dict(type=typ, tagged=side is not None, hit=len(inside) > 0,
                       agree=(pred == side) if (side and pred) else np.nan,
                       d_share=float(np.mean(dead[s:e])) if e > s else np.nan))
        if typ.lower() in ("corner kick", "goal kick") and side:
            bx = atd["ball_x"][s:e] - hl
            bx = bx[np.isfinite(bx)]
            if len(bx):
                end = np.sign(np.median(bx))
            else:
                px = atd["px"][s:e][:, ~gk] - hl
                end = np.sign(np.nanmedian(px))
            home_plus = home_plus_1st if k == 0 else not home_plus_1st
            attacker = "H" if (end > 0) == home_plus else "A"
            geo_team = attacker if typ.lower() == "corner kick" else ("A" if attacker == "H" else "H")
            geo.append(dict(type=typ, agree=geo_team == side, from_ball=len(bx) > 0))

    g = pd.DataFrame(geo)
    print("\n1. set-piece team tag vs geometry (label-free):")
    for typ, x in g.groupby("type"):
        print(f"   {typ:<12} {int(x.agree.sum())}/{len(x)} agree  (end from ball in {int(x.from_ball.sum())})")
    print(f"   all          {int(g.agree.sum())}/{len(g)} = {g.agree.mean():.0%}")

    s = pd.DataFrame(sp)
    print("\n2. set pieces in graded windows vs our predictions:")
    print(f"   {'type':<22} {'n':>3} {'live start in tag':>18} {'mean D share in tag':>20} {'our team = tag':>15}")
    for typ, x in s.groupby("type"):
        ag = x.agree.dropna()
        print(f"   {typ:<22} {len(x):>3} {x.hit.mean():>18.0%} {x.d_share.mean():>20.0%} "
              + (f"{int(ag.sum()):>8}/{len(ag)}" if len(ag) else f"{'-':>15}"))
    ag = s.agree.dropna()
    print(f"   team-tagged set pieces with a live start in the tag: {len(ag)}; our team agrees on "
          f"{int(ag.sum())} ({ag.mean():.0%})")

    # 3: phases
    print("\n3. phases: share of our predicted live frames inside the tag that carry the tagged team")
    rows = []
    for t in tags:
        code = (t["code"] or "").strip()
        side = side_of(t["tags"].get("Team"), names)
        if code.lower() == "set pieces" or side is None or graded_half(t["start_frame"]) is None:
            continue
        seg = labels[t["start_frame"]:t["end_frame"]]
        live = np.isin(seg, ["H", "A"])
        if live.sum() == 0:
            continue
        rows.append(dict(code=code, frames=int(live.sum()), same=int((seg[live] == side).sum()),
                         d_share=float(np.mean(seg == "D"))))
    r = pd.DataFrame(rows).groupby("code")[["frames", "same"]].sum()
    n = pd.DataFrame(rows).groupby("code").size()
    r["share"] = r.same / r.frames
    for code in r.sort_values("share", ascending=False).index:
        print(f"   {code:<32} n {n[code]:>3}  live s {r.frames[code] / FPS:>6.0f}  tagged team {r.share[code]:.0%}")
    print(f"\npredicted live stretches in graded windows: {len(starts)}")


if __name__ == "__main__":
    main()
