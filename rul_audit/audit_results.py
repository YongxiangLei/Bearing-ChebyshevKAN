"""Independent audit of XJTU RUL split, training scaler, checkpoints and scores."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def errmetric(y, p):
    e = np.asarray(p) - np.asarray(y)
    return {'mae_min':np.mean(np.abs(e)), 'rmse_min':np.sqrt(np.mean(e**2)),
            'mean_overestimate_min':np.maximum(e,0).mean(), 'overestimate_rate':np.mean(e>0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('audit_report.json'))
    p.add_argument('--results-name', default='results')
    p.add_argument('--allow-incomplete', action='store_true', help='Permit smoke-test output; mark report incomplete')
    args = p.parse_args()
    study = args.study.resolve()
    sys.path.insert(0, str(study))
    from build_features import EXPECTED_COUNTS, FEATURES, CONDITIONS
    results = study/args.results_name
    data = pd.read_csv(study/'data'/'features_and_labels.csv')
    pred = pd.read_csv(results/'predictions.csv')
    folds = json.loads((results/'folds.json').read_text(encoding='utf8'))
    summary = json.loads((results/'summary.json').read_text(encoding='utf8'))
    models = ('age_ridge','features_ridge','mlp','kan')
    seeds = summary['training_seeds']
    test_keys={(f['test_bearing'],f['training_seed']) for f in folds}
    if not args.allow_incomplete:
        assert len(folds)==15*len(seeds) and len(pred)==len(data)*len(seeds)
    assert len(folds)==len(test_keys) and len(pred)==sum(EXPECTED_COUNTS[b] for b,s in test_keys)
    assert not pred.duplicated(['bearing','record','seed']).any()
    assert {(b,int(s)) for b,s in zip(pred.bearing,pred.seed)}==test_keys
    assert not set(summary['input_columns']) & {'rul_minutes','record','source_file','bearing','condition'}
    assert summary['input_columns']==FEATURES+['elapsed_minutes','speed_hz','load_kN']
    check = data[['bearing','record','rul_minutes','elapsed_minutes']].merge(pred[['bearing','record','seed','rul_minutes','elapsed_minutes']],on=['bearing','record'],how='inner',validate='one_to_many',suffixes=('_data','_pred'))
    assert len(check)==len(pred)
    assert np.allclose(check.rul_minutes_data,check.rul_minutes_pred)
    assert np.allclose(check.elapsed_minutes_data,check.elapsed_minutes_pred)
    for b, g in data.groupby('bearing'):
        assert len(g)==EXPECTED_COUNTS[b]
        assert np.array_equal(np.sort(g.record.to_numpy()),np.arange(1,len(g)+1))
        assert np.allclose(g.rul_minutes,len(g)-g.record)
        assert np.allclose(g.elapsed_minutes,g.record-1)
        assert g.condition.nunique()==1
    sys.path.insert(0,str(args.project.resolve()/'cwru'/'exp'))
    from adaptive_chebyshev_kan import AdaptiveChebyshevKANLayer
    from torch import nn
    class KAN(nn.Module):
        def __init__(self, dim, width):
            super().__init__(); self.l1=AdaptiveChebyshevKANLayer(dim,width,3); self.l2=AdaptiveChebyshevKANLayer(width,1,3)
        def forward(self,x): return self.l2(self.l1(x)).squeeze(1)
    class MLP(nn.Module):
        def __init__(self,dim,width):
            super().__init__(); self.net=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Linear(width,1))
        def forward(self,x): return self.net(x).squeeze(1)
    torch.set_num_threads(2)
    max_checkpoint_difference={m:0. for m in models}
    max_scaler_difference=0.
    max_metric_difference=0.
    recomputed={m:[] for m in models}
    seen=set()
    for fold in folds:
        b=fold['test_bearing']; seed=fold['training_seed']; key=(b,seed)
        assert key not in seen; seen.add(key)
        train=set(fold['train_bearings']); val=set(fold['validation_bearings']); test={b}
        assert len(train)==11 and len(val)==3 and not train&val and not (train|val)&test
        assert train|val|test==set(EXPECTED_COUNTS)
        assert set(fold['scaler_fit_bearings'])==train
        assert set(data[data.bearing.isin(val)].condition)==set(CONDITIONS)
        c=torch.load(results/'checkpoints'/f'{b}_seed{seed}.pt',map_location='cpu',weights_only=False)
        assert set(c['train_bearings'])==train and set(c['validation_bearings'])==val and c['test_bearing']==b
        assert c['input_columns']==summary['input_columns'] and c['training_seed']==seed
        tr=data[data.bearing.isin(train)]; trids=tr.bearing.to_numpy(); _, inverse, counts=np.unique(trids,return_inverse=True,return_counts=True); w=1/counts[inverse]; w/=w.mean()
        scaler=StandardScaler().fit(tr[summary['input_columns']].to_numpy(float),sample_weight=w)
        target=tr.rul_minutes.to_numpy(float); target_mean=np.average(target,weights=w); target_std=np.sqrt(np.average((target-target_mean)**2,weights=w))+1e-8
        diff=max(np.max(np.abs(c['scaler_mean']-scaler.mean_)),np.max(np.abs(c['scaler_scale']-scaler.scale_)))
        max_scaler_difference=max(max_scaler_difference,float(diff)); assert diff < 1e-7
        tg=data[data.bearing==b].sort_values('record'); pr=pred[(pred.bearing==b)&(pred.seed==seed)].sort_values('record')
        assert len(tg)==len(pr)==EXPECTED_COUNTS[b] and np.array_equal(tg.record,pr.record)
        X=scaler.transform(tg[summary['input_columns']].to_numpy(float)); y=tg.rul_minutes.to_numpy()
        for m in models:
            saved=c['models'][m]
            if m.endswith('ridge'):
                out=np.maximum(X[:,saved['columns']]@saved['coef']+saved['intercept'],0)
            else:
                assert abs(saved['target_train_mean']-target_mean)<1e-7
                assert abs(saved['target_train_std']-target_std)<1e-7
                net=(KAN if m=='kan' else MLP)(X.shape[1],saved['width'])
                net.load_state_dict(saved['state_dict']); net.eval()
                with torch.no_grad():
                    out=np.maximum(net(torch.tensor(X,dtype=torch.float32)).numpy()*saved['target_train_std']+saved['target_train_mean'],0)
            diff=float(np.max(np.abs(out-pr[m+'_pred_min'].to_numpy())))
            max_checkpoint_difference[m]=max(max_checkpoint_difference[m],diff); assert diff<2e-3,(key,m,diff)
            mm=errmetric(y,pr[m+'_pred_min'].to_numpy())
            for name,v in mm.items():
                delta=abs(v-fold['test_metrics'][m][name]); max_metric_difference=max(max_metric_difference,float(delta)); assert delta<2e-4,(key,m,name,delta)
            assert np.array_equal(pr['late_last_20pct_evaluation_only'].to_numpy(),y<=.2*(len(y)-1))
            mm['late_last_20pct_macro']=errmetric(y[y<=.2*(len(y)-1)],pr.loc[y<=.2*(len(y)-1),m+'_pred_min'].to_numpy())
            recomputed[m].append(mm)
    assert len(seen)==len(folds)
    if not args.allow_incomplete: assert seen=={(b,s) for b in EXPECTED_COUNTS for s in seeds}
    for m in models:
        for metric in ('mae_min','rmse_min','mean_overestimate_min','overestimate_rate'):
            got=np.mean([f[metric] for f in recomputed[m]])
            assert abs(got-summary['macro_results'][m][metric])<2e-4,(m,metric,got)
            got=np.mean([f['late_last_20pct_macro'][metric] for f in recomputed[m]])
            assert abs(got-summary['macro_results'][m]['late_last_20pct_macro'][metric])<2e-4,(m,metric,'late',got)
        for field,fold_key in [('mean_mae_by_training_seed','training_seed'),('mean_mae_by_condition','test_condition'),('mean_mae_by_test_bearing','test_bearing')]:
            if field in summary['macro_results'][m]:
                for label,expected in summary['macro_results'][m][field].items():
                    got=np.mean([f['test_metrics'][m]['mae_min'] for f in folds if str(f[fold_key])==label])
                    assert abs(got-expected)<2e-4,(m,field,label,got,expected)
    result={'status':'passed','complete_15_bearings':len(seen)==15*len(seeds),'independent_test_bearings':len(set(b for b,s in seen)),'seeds':seeds,'folds':len(seen),'records_per_full_seed':len(data),
            'verified':['target and elapsed labels','split disjointness','one validation bearing per condition','training-only bearing-weighted input and target scalers',
                        'saved checkpoint vs predictions','per-fold and per-bearing-averaged metrics','late-mask evaluation only'],
            'max_abs_scaler_difference':max_scaler_difference,'max_abs_checkpoint_prediction_difference':max_checkpoint_difference,
            'max_abs_metric_difference':max_metric_difference,
            'limitations':['Static audit and checkpoint reconstruction cannot prove absence of all data leakage or generalization to new conditions.',
                           'The three model seeds do not constitute extra independent bearing observations.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
