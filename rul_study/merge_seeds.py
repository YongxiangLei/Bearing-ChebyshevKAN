"""Merge completed single-seed runs; fail rather than report partial CV."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from build_features import EXPECTED_COUNTS, CONDITIONS

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).parent)
    ap.add_argument('--seeds',default='42,43,44')
    ap.add_argument('--output',type=Path,default=Path(__file__).parent/'results_merged')
    args=ap.parse_args(); seeds=[int(s) for s in args.seeds.split(',')]
    folds=[]; preds=[]; base=None; split_by_bearing={}; checkpoints=[]
    for seed in seeds:
        directory=args.root/f'results_seed{seed}'
        summary=json.loads((directory/'summary.json').read_text(encoding='utf8'))
        f=json.loads((directory/'folds.json').read_text(encoding='utf8'))
        p=pd.read_csv(directory/'predictions.csv')
        if summary['training_seeds'] != [seed] or not summary['complete_15_folds'] or len(f)!=15:
            raise ValueError(f'Incomplete run or seed mismatch: {directory}')
        if len(p)!=sum(EXPECTED_COUNTS.values()) or set(p.seed)!={seed}:
            raise ValueError(f'Incorrect prediction count/seed: {directory}')
        if p.duplicated(['bearing','record']).any() or set(p.bearing.unique())!=set(EXPECTED_COUNTS):
            raise ValueError(f'Duplicate or missing bearing predictions: {directory}')
        for b,n in EXPECTED_COUNTS.items():
            if int((p.bearing==b).sum())!=n:raise ValueError(f'Bad per-bearing count: {seed}/{b}')
        if len({x['test_bearing'] for x in f})!=15: raise ValueError('Missing or duplicate fold')
        for entry in f:
            b=entry['test_bearing']; split=(tuple(entry['train_bearings']),tuple(entry['validation_bearings']))
            if entry['training_seed']!=seed: raise ValueError('Incorrect fold seed')
            if b in split_by_bearing and split_by_bearing[b]!=split: raise ValueError(f'Split differs between seeds for {b}')
            split_by_bearing[b]=split
            ckpt=directory/'checkpoints'/f'{b}_seed{seed}.pt'
            if not ckpt.exists():raise ValueError(f'Missing {ckpt}')
            checkpoints.append(ckpt)
        if base is None:base=summary
        else:
            for key in ('split_seed','input_columns','age_baseline_columns','epochs_max','patience','candidate_budget','model_note','target'):
                if base[key]!=summary[key]:raise ValueError(f'Run settings differ: {key}')
        folds.extend(f); preds.append(p)
    if len(folds)!=15*len(seeds): raise ValueError('Wrong number of folds')
    aggregate={}
    for name in ('age_ridge','features_ridge','mlp','kan'):
        q=[f['test_metrics'][name] for f in folds]
        result={k:float(np.mean([m[k] for m in q])) for k in ('mae_min','rmse_min','mean_overestimate_min','overestimate_rate')}
        result['late_last_20pct_macro']={k:float(np.mean([m['late_last_20pct'][k] for m in q])) for k in ('mae_min','rmse_min','mean_overestimate_min','overestimate_rate')}
        result['mean_mae_by_training_seed']={str(s):float(np.mean([f['test_metrics'][name]['mae_min'] for f in folds if f['training_seed']==s])) for s in seeds}
        result['mean_mae_by_condition']={c:float(np.mean([f['test_metrics'][name]['mae_min'] for f in folds if f['test_condition']==c])) for c in CONDITIONS}
        result['mean_mae_by_test_bearing']={b:float(np.mean([f['test_metrics'][name]['mae_min'] for f in folds if f['test_bearing']==b])) for b in sorted(EXPECTED_COUNTS)}
        aggregate[name]=result
    base.update({'training_seeds':seeds,'complete_15_folds':True,'folds_completed':len(folds),'macro_results':aggregate,
                 'aggregation_note':'Mean of per-bearing metrics, then mean over training seeds; seeds share the same test bearings and validation splits, and are not independent bearings.'})
    args.output.mkdir(parents=True,exist_ok=True); (args.output/'checkpoints').mkdir(exist_ok=True)
    for ckpt in checkpoints: shutil.copy2(ckpt,args.output/'checkpoints'/ckpt.name)
    pd.concat(preds,ignore_index=True).to_csv(args.output/'predictions.csv',index=False)
    (args.output/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf8')
    (args.output/'summary.json').write_text(json.dumps(base,indent=2,ensure_ascii=False),encoding='utf8')
    print('Merged',len(folds),'test folds and',sum(len(p) for p in preds),'predictions')
    for name,metrics in aggregate.items():print(name,metrics['mae_min'],metrics['rmse_min'])

if __name__=='__main__':main()
