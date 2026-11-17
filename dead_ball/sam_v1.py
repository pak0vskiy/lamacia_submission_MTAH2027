# Vendored verbatim from origin/sam_v1:lamacia_possession_v1.py (commit 407143d, LaMacIA v1 by Sam).
# Do not edit behaviour here; dead_ball/v1_adapter.py feeds it IDSSE data.
"""Interpretable possession baseline. Run from the challenge repository root.
No GT is read. Threshold defaults are provisional, not learned or calibrated.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

@dataclass
class Config:
    fps: float = 25.0
    n_frames: int = 199319
    length: float = 105.3
    width: float = 68.0
    motion_window_s: float = 1.0
    min_players: int = 12
    control_radius_m: float = 5.0
    margin_scale_m: float = 1.5
    ownership_threshold: float = 0.35
    slow_speed_mps: float = 0.9
    fast_speed_mps: float = 1.8
    stationary_speed_mps: float = 0.6
    edge_distance_m: float = 2.0
    dead_enter_threshold: float = 0.8
    dead_exit_threshold: float = 0.5
    ownership_confirm_s: float = 0.6
    dead_enter_confirm_s: float = 2.0
    dead_exit_confirm_s: float = 0.6
    max_motion_gap_s: float = 0.12
    # Inclusive, organiser-published evaluation boundaries; not inferred possession labels.
    halves: tuple = ((11185, 90651), (110808, 190197))


def team_slots(meta):
    root = ET.parse(meta).getroot()
    teams = [x.get('id') for x in root.iter('Team')]
    if len(teams) != 2:
        raise ValueError('Expected two Team entries in metadata.')
    players = list(root.iter('Player'))
    ids = [x.get('teamId') for x in players]
    if not ids or any(x not in teams for x in ids):
        raise ValueError('Missing or unknown track-slot team membership.')
    return np.array([0 if x == teams[0] else 1 for x in ids])


def parse_group(group):
    values = [float(x) for x in group.strip().split(',')]
    if len(values) != 3:
        raise ValueError('Expected x,y,speed per observation.')
    return values


def load_features(meta, raw, c):
    sides = team_slots(meta)
    rows = np.full((c.n_frames, 9), np.nan)
    seen = np.zeros(c.n_frames, dtype=bool)
    scale = np.array([c.length, c.width])
    with open(raw, encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                frame, groups, ball_group = line.strip().split(':')
                i = int(frame) - 1
                if not 0 <= i < c.n_frames or seen[i]:
                    raise ValueError('Duplicate or out-of-range frame ID.')
                p = np.array([parse_group(g) for g in groups.split(';')])
                if len(p) != len(sides):
                    raise ValueError('Track-slot count differs from metadata.')
                ball = np.array(parse_group(ball_group))
                valid = np.isfinite(p[:, :2]).all(axis=1)
                speeds = p[valid & np.isfinite(p[:, 2]), 2]
                speeds = speeds[speeds >= 0]
                rows[i, 0] = np.median(speeds) if len(speeds) else np.nan
                rows[i, 1] = len(speeds)
                rows[i, 2:4] = ball[:2] * scale
                rows[i, 4] = float(np.isfinite(ball[:2]).all())
                if rows[i, 4]:
                    dist = np.linalg.norm((p[:, :2] - ball[:2]) * scale, axis=1)
                    for side in (0, 1):
                        available = dist[valid & (sides == side)]
                        rows[i, 5 + side] = available.min() if len(available) else np.nan
                    x, y = rows[i, 2:4]
                    rows[i, 7] = min(x, c.length-x, y, c.width-y)
                seen[i] = True
            except (ValueError, IndexError) as exc:
                raise ValueError(f'Invalid raw ATD line {line_no}: {exc}') from exc
    if not seen.all():
        raise ValueError(f'Expected all {c.n_frames} frames; missing {int((~seen).sum())}.')
    return pd.DataFrame(rows[:, :8], columns=[
        'median_speed', 'n_speed', 'ball_x_m', 'ball_y_m', 'ball_visible',
        'd_home_m', 'd_away_m', 'edge_m'])


def evidence(f, c):
    """Signed ownership evidence and uncalibrated [0,1] dead/live evidence.
    Missing evidence stays NaN, never silently becomes a negative observation.
    Calculations reset at each half; no interpolation or halftime leakage.
    """
    n = len(f)
    owner = np.full(n, np.nan)
    dead = np.full(n, np.nan)
    live = np.full(n, np.nan)
    for start, end in c.halves:
        if not 0 <= start <= end < n:
            raise ValueError('Half boundaries outside frame range.')
        block = f.iloc[start:end+1]
        speed = block.median_speed.where(block.n_speed >= c.min_players)
        window = max(1, round(c.motion_window_s*c.fps))
        motion = speed.rolling(window, min_periods=max(1, window//2)).median()
        motion = motion.where(speed.notna()).to_numpy()
        slow = np.clip((c.fast_speed_mps-motion)/
                       (c.fast_speed_mps-c.slow_speed_mps), 0, 1)
        moving = 1-slow
        dh, da = block.d_home_m.to_numpy(), block.d_away_m.to_numpy()
        proximity = np.clip(1-np.minimum(dh, da)/c.control_radius_m, 0, 1)
        own = np.tanh((da-dh)/c.margin_scale_m)*proximity
        visible = block.ball_visible.to_numpy().astype(bool)
        xy = block[['ball_x_m', 'ball_y_m']].to_numpy()
        ball_speed = np.full(len(block), np.nan)
        last = None
        for j in range(len(block)):
            if visible[j]:
                if last is not None and (j-last)/c.fps <= c.max_motion_gap_s:
                    ball_speed[j] = np.linalg.norm(xy[j]-xy[last])*c.fps/(j-last)
                last = j
        stationary = np.isfinite(ball_speed) & (ball_speed <= c.stationary_speed_mps)
        at_edge = block.edge_m.to_numpy() <= c.edge_distance_m
        outside = block.edge_m.to_numpy() < 0
        # Missing ball alone is NOT dead; collective slowdown is always required.
        cue = (~visible) | (stationary & at_edge) | outside
        d = slow*cue.astype(float)
        # Restart: collective acceleration plus near-ball team evidence.
        l = moving*proximity
        sl = slice(start, end+1)
        owner[sl], dead[sl], live[sl] = own, d, l
    return owner, dead, live


def decode(owner, dead, live, c):
    labels = np.full(len(owner), 'X', dtype='<U1')
    thresholds = {
        'owner': max(1, round(c.ownership_confirm_s*c.fps)),
        'dead': max(1, round(c.dead_enter_confirm_s*c.fps)),
        'live': max(1, round(c.dead_exit_confirm_s*c.fps))}
    for start, end in c.halves:
        state, candidate, count = 'X', None, 0
        for i in range(start, end+1):
            proposed, kind = None, None
            team = ('H' if owner[i] > 0 else 'A') if (
                np.isfinite(owner[i]) and abs(owner[i]) >= c.ownership_threshold) else None
            if state == 'D':
                if np.isfinite(live[i]) and live[i] >= c.dead_exit_threshold and team:
                    proposed, kind = team, 'live'
            elif np.isfinite(dead[i]) and dead[i] >= c.dead_enter_threshold:
                proposed, kind = 'D', 'dead'
            elif team and team != state:
                proposed, kind = team, 'owner'
            key = (proposed, kind) if proposed else None
            if key is None:
                candidate, count = None, 0
            else:
                count = count+1 if key == candidate else 1
                candidate = key
                if count >= thresholds[kind]:
                    state = proposed
                    candidate, count = None, 0
            # Causal confirmation: do not backdate transitions or invent gap boundaries.
            labels[i] = state
    return labels


def self_test():
    c = Config(fps=10, n_frames=60, halves=((0, 59),),
               ownership_confirm_s=.2, dead_enter_confirm_s=.3,
               dead_exit_confirm_s=.2)
    owner = np.full(60, .9); dead = np.zeros(60); live = np.ones(60)
    owner[10] = -.9
    owner[20:30] = np.nan
    y = decode(owner, dead, live, c)
    assert y[10] == 'H' and np.all(y[20:30] == 'H')
    dead[30:40] = 1; live[30:40] = 0
    owner[40:] = -.9
    y = decode(owner, dead, live, c)
    assert y[32] == 'D' and y[41] == 'A'
    c.halves = ((0, 19), (40, 59))
    y = decode(owner, dead, live, c)
    assert np.all(y[20:40] == 'X') and y[40] == 'X'
    f = pd.DataFrame({'median_speed': np.full(60, 2.2), 'n_speed': 22,
                      'ball_x_m': np.nan, 'ball_y_m': np.nan,
                      'ball_visible': 0., 'd_home_m': np.nan,
                      'd_away_m': np.nan, 'edge_m': np.nan})
    c.halves = ((0,59),)
    _, d, _ = evidence(f, c)
    assert not np.any(d[np.isfinite(d)] > 0)
    f['median_speed'] = .4
    _, d, _ = evidence(f, c)
    assert np.nanmax(d) == 1
    f['n_speed'] = 2
    _, d, _ = evidence(f, c)
    assert np.isnan(d).all()
    print('Synthetic tests passed: flicker, gaps, dead/restart, half reset, evidence guards.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta', type=Path)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, default=Path('evaluation/results/lamacia_v1.csv'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.meta is None or args.raw is None:
        parser.error('--meta and --raw are required unless --self-test is used')
    values = json.loads(args.config.read_text()) if args.config else {}
    c = Config(**values)
    if not (c.fast_speed_mps > c.slow_speed_mps >= 0 and c.margin_scale_m > 0
            and c.control_radius_m > 0 and c.fps > 0):
        raise ValueError('Invalid configuration scales.')
    f = load_features(args.meta, args.raw, c)
    owner, dead, live = evidence(f, c)
    labels = decode(owner, dead, live, c)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'frame': np.arange(c.n_frames), 'label': labels}).to_csv(
        args.out, index=False, lineterminator='\n')
    # Aggregates only: never export frame-level input features by default.
    summary = {'config': asdict(c), 'label_counts': {
        x: int(np.sum(labels == x)) for x in 'HADX'},
        'graded_state_changes': int(sum(np.sum(labels[a+1:b+1] != labels[a:b])
                                       for a,b in c.halves)),
        'warning': 'Uncalibrated evidence and provisional defaults; no GT used.'}
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
