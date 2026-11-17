#!/usr/bin/env python3
"""Two-stage possession: fixed seven-fold D ensemble, then independent H/A decoder.
Place in project root beside games.py. Uses original ATD adapter and trained models.
No previous possession CSV, competition GT, video, new training, or GPU required.
"""
import argparse
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )

ROOT = Path(__file__).resolve().parent.parent
MATCH_IDS = ('J03WMX','J03WN1','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9')
MODEL_NAME = 'model-no-ball_spd_1@mixed'


@dataclass
class OwnerConfig:
    # Provisional design parameters, NOT selected using competition labels.
    control_radius_m: float = 4.0
    margin_scale_m: float = 1.5
    min_team_margin_m: float = 0.3
    evidence_threshold: float = 0.25
    initial_confirm_s: float = 0.12
    switch_confirm_s: float = 0.24
    restart_lookahead_s: float = 2.0
    # Offline: backdate a confirmed switch to its first supporting observation.
    backdate_switches: bool = True

    def validate(self):
        values = asdict(self)
        for key, value in values.items():
            if key == 'backdate_switches':
                if type(value) is not bool:
                    raise ValueError('backdate_switches must be boolean')
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.control_radius_m <= 0 or self.margin_scale_m <= 0:
            raise ValueError('Distance scales must be positive')
        if not 0 < self.evidence_threshold <= 1:
            raise ValueError('evidence_threshold must be in (0,1]')


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask,dtype=np.int8), 0]))
    return list(zip(edges[::2],edges[1::2]))


def ownership_evidence(period, config):
    """H positive, A negative; metres, centre-origin; missing evidence remains NaN."""
    ball = np.asarray(period.ball)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError('Expected ball shape (frames,2)')
    n = len(ball)
    team_distances = []
    for xy in (period.home, period.away):
        xy = np.asarray(xy)
        if xy.ndim != 3 or xy.shape[0] != n or xy.shape[2] != 2:
            raise ValueError('Expected player shape (frames,slots,2)')
        if xy.shape[1] == 0:
            best = np.full(n,np.nan)
        else:
            dist = np.linalg.norm(xy-ball[:,None,:],axis=2)
            dist[~np.isfinite(dist)] = np.inf
            best = dist.min(axis=1)
            best[~np.isfinite(best)] = np.nan
        team_distances.append(best)
    dh, da = team_distances
    margin = da-dh
    proximity = np.clip(1-np.minimum(dh,da)/config.control_radius_m,0,1)
    signed = np.tanh(margin/config.margin_scale_m)*proximity
    ambiguous = np.abs(margin) < config.min_team_margin_m
    signed[ambiguous] = 0.0
    return signed


def decode_owners(evidence, dead_mask, fps, config):
    """Decode each live interval independently; D is supplied, never inferred here.

    Unknown observations hold initialized ownership. They reset any unconfirmed
    candidate. A restart can fill its leading unknown prefix only when ownership
    is confirmed within the bounded look-ahead. Later initialization backdates
    only to its own first supporting observation. No state crosses a D interval.
    """
    evidence = np.asarray(evidence,float)
    dead_mask = np.asarray(dead_mask,bool)
    if evidence.ndim != 1 or evidence.shape != dead_mask.shape:
        raise ValueError('Evidence and dead mask must be aligned vectors')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive')
    out = np.full(len(evidence),'X',dtype='<U1')
    out[dead_mask] = 'D'
    init_frames = max(1,round(config.initial_confirm_s*fps))
    switch_frames = max(1,round(config.switch_confirm_s*fps))
    lookahead_frames = max(0,round(config.restart_lookahead_s*fps))
    stats = {'live_intervals':0,'initialized_intervals':0,
             'entirely_unknown_intervals':0,'confirmed_team_switches':0,
             'restart_prefix_frames_filled':0}
    for start,stop in runs(~dead_mask):
        stats['live_intervals'] += 1
        state = None
        candidate = None
        candidate_start = -1
        count = 0
        for i in range(start,stop):
            value = evidence[i]
            proposed = None
            if np.isfinite(value) and abs(value) >= config.evidence_threshold:
                proposed = 'H' if value > 0 else 'A'
            if proposed is None or proposed == state:
                candidate = None
                count = 0
            else:
                if proposed == candidate:
                    count += 1
                else:
                    candidate = proposed
                    candidate_start = i
                    count = 1
                required = init_frames if state is None else switch_frames
                if count >= required:
                    if state is None:
                        stats['initialized_intervals'] += 1
                        fill_start = candidate_start
                        # Confirmation must occur within the initial time bound.
                        if i-start <= lookahead_frames:
                            fill_start = start
                            stats["restart_prefix_frames_filled"] += int(candidate_start - start)
                        out[fill_start:i+1] = proposed
                    else:
                        stats['confirmed_team_switches'] += 1
                        if config.backdate_switches:
                            out[candidate_start:i+1] = proposed
                    state = proposed
                    candidate = None
                    count = 0
            if state is not None:
                out[i] = state
        if state is None:
            stats['entirely_unknown_intervals'] += 1
    if not np.array_equal(out=='D',dead_mask):
        raise AssertionError('Ownership decoder changed the supplied dead mask')
    stats['unknown_live_frames'] = int(np.sum((out=='X')&~dead_mask))
    return out,stats


def average_dead_probability(folds, features):
    predictions=[]
    for fold in folds:
        clf=fold['fitted']
        cols=list(clf.feature_names_in_)
        if set(cols)-set(features.columns):
            raise ValueError(f'Missing detector features: {set(cols)-set(features.columns)}')
        classes=list(clf.classes_)
        if 1 not in classes:
            raise ValueError('No dead-positive class 1 in classifier')
        p=np.asarray(clf.predict_proba(features[cols])[:,classes.index(1)],float)
        if p.shape!=(len(features),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid dead probabilities')
        predictions.append(p)
    return np.mean(predictions,axis=0)


def load_folds(folder):
    paths=[folder/f'fold_{MODEL_NAME}_{mid}.joblib' for mid in MATCH_IDS]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing trained models:\n'+'\n'.join(missing))
    # joblib/pickle can execute code: load only trusted locally trained files.
    folds=[joblib.load(path) for path in paths]
    schemas=[]
    for fold in folds:
        if not isinstance(fold,dict) or not {'fitted','params'}.issubset(fold):
            raise ValueError('Expected fold dictionary with fitted and params')
        clf=fold['fitted']
        if not hasattr(clf,'feature_names_in_'):
            raise ValueError('Model has no named feature schema')
        cols=list(clf.feature_names_in_)
        if 'ball_spd_1' in cols:
            raise ValueError('Wrong model variant: ball_spd_1 must be excluded')
        schemas.append(cols)
    if any(x!=schemas[0] for x in schemas[1:]):
        raise ValueError('Fold feature schemas differ')
    votes=Counter(tuple(sorted(x['params'].items())) for x in folds)
    selected,vote_count=votes.most_common(1)[0]
    if sum(count==vote_count for count in votes.values())>1:
        warnings.warn('Decoder vote tied; reference first-seen tie break with sorted match IDs is used')
    decoder=dict(selected)
    if not {'enter','exit','min_dead_s'}.issubset(decoder):
        raise ValueError('Incomplete decoder parameters')
    if not 0<=decoder['exit']<=decoder['enter']<=1:
        raise ValueError('Invalid decoder hysteresis thresholds')
    if decoder['min_dead_s']<0 or decoder.get('min_alive_s',1.)<0:
        raise ValueError('Invalid decoder durations')
    return folds,paths,decoder,votes,schemas[0]


def load_adapter(path):
    if not path.is_file():
        raise FileNotFoundError(f'ATD adapter not found: {path}')
    spec=importlib.util.spec_from_file_location('two_stage_atd_adapter',path)
    adapter=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=adapter
    spec.loader.exec_module(adapter)
    return adapter


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def self_test():
    from types import SimpleNamespace
    c=OwnerConfig(initial_confirm_s=.2,switch_confirm_s=.3,restart_lookahead_s=.5)
    fps=10.
    # Short opposing flicker is ignored; supported switches backdate to onset.
    evidence=np.full(20,.8)
    evidence[5]=-.8
    evidence[10:]=-.8
    y,s=decode_owners(evidence,np.zeros(20,bool),fps,c)
    assert np.all(y[:10]=='H') and np.all(y[10:]=='A')
    # A short initial blackout can be filled after bounded confirmation.
    evidence=np.r_[np.nan,np.nan,.8,.8,.8]
    y,_=decode_owners(evidence,np.zeros(5,bool),fps,c)
    assert np.all(y=='H')
    # Long initial blackout stays X; no unbounded backfill.
    evidence=np.r_[np.full(10,np.nan),.8,.8,.8]
    y,_=decode_owners(evidence,np.zeros(13,bool),fps,c)
    assert np.all(y[:10]=='X') and np.all(y[10:]=='H')
    # No ownership leaks across a dead interval, even with no restart evidence.
    evidence=np.r_[np.full(5,.8),np.full(2,.8),np.full(5,np.nan)]
    dead=np.zeros(12,bool);dead[5:7]=True
    y,_=decode_owners(evidence,dead,fps,c)
    assert np.all(y[:5]=='H') and np.all(y[5:7]=='D') and np.all(y[7:]=='X')
    # Missing ball during initialized live play holds ownership.
    evidence=np.r_[.8,.8,np.full(10,np.nan)]
    y,_=decode_owners(evidence,np.zeros(12,bool),fps,c)
    assert np.all(y=='H')
    y,_=decode_owners(np.full(5,np.nan),np.ones(5,bool),fps,c)
    assert np.all(y=='D')
    y,_=decode_owners(np.full(5,np.nan),np.zeros(5,bool),fps,c)
    assert np.all(y=='X')
    period=SimpleNamespace(ball=np.array([[0.,0.],[0.,0.],[np.nan,np.nan]]),
        home=np.array([[[.2,0.]],[[3.,0.]],[[.2,0.]]]),
        away=np.array([[[3.,0.]],[[.2,0.]],[[3.,0.]]]))
    e=ownership_evidence(period,c)
    assert e[0]>.25 and e[1]<-.25 and np.isnan(e[2])
    class Dummy:
        feature_names_in_=np.array(['a']);classes_=np.array([False,True])
        def __init__(self,p): self.p=p
        def predict_proba(self,x): return np.tile([1-self.p,self.p],(len(x),1))
    pr=average_dead_probability([{'fitted':Dummy(.2)},{'fitted':Dummy(.8)}],pd.DataFrame({'a':[0,1]}))
    assert np.allclose(pr,.5)
    print('All synthetic tests passed: evidence, ensemble average, persistence, '
          'switch confirmation, restart look-ahead, D resets, and unknowns.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models-dir',type=Path,default=ROOT/'artifacts/dead_ball')
    p.add_argument('--adapter',type=Path,default=ROOT/'analysis/08_atd_vs_idsse_features.py')
    p.add_argument('--owner-config',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'evaluation/results/LaMacIA_two_stage.csv')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test();return
    config=OwnerConfig(**(json.loads(args.owner_config.read_text()) if args.owner_config else {}))
    config.validate()
    if args.out.suffix.lower()!='.csv':
        p.error('--out must be a .csv file')
    meta_path=args.out.with_suffix('.metadata.json')
    if not args.overwrite and (args.out.exists() or meta_path.exists()):
        p.error('Output or metadata already exists; use --overwrite intentionally')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))
    from games import GAME,GAMES,half_video_blocks
    from dead_ball.features import frame_features
    from dead_ball.pipeline import decode_dead
    import possession_metrics as pm
    fps=float(GAMES[GAME]['fps']);n=GAMES[GAME]['n_video_frames']
    if abs(fps-25.)>1e-6:
        raise ValueError('Original ATD adapter assumes 25 fps')
    folds,paths,decoder,votes,features=load_folds(args.models_dir)
    print('Loaded 7 selected dead-ball models. Decoder:',decoder)
    adapter=load_adapter(args.adapter)
    blocks=half_video_blocks(GAME)
    if list(adapter.GRADED)!=[(a,b+1) for _,a,b in blocks]:
        raise ValueError('Adapter timeline differs from games.py')
    print('Loading tracking through original ATD adapter...')
    periods,hl,hw=adapter.atd_periods()
    if len(periods)!=len(blocks):
        raise ValueError('Unexpected number of periods')
    if not np.allclose([2*hl,2*hw],GAMES[GAME]['pitch'],atol=.01):
        warnings.warn('Cached pitch differs from games.py; using adapter dimensions consistently')
    labels=np.full(n,'X',dtype='<U1');active=np.zeros(n,bool);half_stats=[]
    for period,(half,a,b) in zip(periods,blocks):
        if len(period)!=b-a+1:
            raise ValueError('ATD period/frame alignment mismatch')
        print(f'Half {half}: extracting D features and estimating ownership...')
        F=frame_features(period,fps,hl,hw)
        probability=average_dead_probability(folds,F)
        dead=decode_dead(probability,fps,**decoder)
        evidence=ownership_evidence(period,config)
        prediction,stats=decode_owners(evidence,dead,fps,config)
        labels[a:b+1]=prediction;active[a:b+1]=True
        stats.update(half=int(half),frames=len(dead),dead_frames=int(dead.sum()),
                     owner_evidence_frames=int(np.sum(np.isfinite(evidence)&(np.abs(evidence)>=config.evidence_threshold))),
                     dead_mask_preserved=bool(np.array_equal(prediction=='D',dead)))
        half_stats.append(stats)
        print(json.dumps(stats, indent=2, default=json_default))
    table=pd.DataFrame({'frame':np.arange(n),'label':labels})
    reason=pm.validate_submission(table,n)
    if reason:
        raise ValueError(f'Official format validation failed: {reason}')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.out,index=False,encoding='utf-8',lineterminator='\n')
    import sklearn
    dependencies={}
    for file in [args.adapter,ROOT/'analysis/atd_cache.py',ROOT/'dead_ball/features.py',
                 ROOT/'dead_ball/pipeline.py',ROOT/'games.py']:
        if file.is_file(): dependencies[str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)]=sha256(file)
    manifest={
        'method':'fixed D ensemble + separate geometry H/A decoder; not trained H/A ML',
        'game':GAME,'fps':fps,'frames':n,'halves_inclusive':blocks,
        'owner_config':asdict(config),'dead_decoder':decoder,
        'decoder_votes':[{'params':dict(k),'votes':v} for k,v in votes.items()],
        'dead_model_features':features,
        'models':[{'file':x.name,'sha256':sha256(x)} for x in paths],
        'script_sha256':sha256(Path(__file__)),
        'dependency_sha256':dependencies,
        'submission_sha256':sha256(args.out),
        'versions':{'python':sys.version,'numpy':np.__version__,
                    'pandas':pd.__version__,'sklearn':sklearn.__version__,
                    'joblib':joblib.__version__},
        'half_statistics':half_stats,
        'graded_counts':{x:int(np.sum(labels[active]==x)) for x in 'HADX'},
        'total_counts':{x:int(np.sum(labels==x)) for x in 'HADX'},
        'warning':'No competition GT read. Owner parameters are unvalidated starting assumptions. '
                  'External validation required; no improvement claimed. Preserve source ATD/cache provenance.'}
    meta_path.write_text(
        json.dumps(manifest, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print('Official format validation passed. Submission:',args.out)
    print('Graded-frame counts:',manifest['graded_counts'])
    print('SHA-256:',manifest['submission_sha256'])
    print('No GT read or training performed. Use the official evaluator separately.')


if __name__=='__main__':
    main()
