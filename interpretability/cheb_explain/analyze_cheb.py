"""Cross-fitted explanation for 45 held-out-bearing Chebyshev KAN checkpoints.

Attributions explain the *clipped* predicted minutes, not the true RUL or a
causal effect.  All backgrounds are the respective training means.  Means are
first calculated per bearing and seed, then weighted equally across folds.
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,default=Path(__file__).parent)
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=256)
    a=parser.parse_args(); root=a.project.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    sys.path.insert(0,str(root/'rul_study'))
    from run_experiments import net_class
    model_cls=net_class(root)['kan']
    source=root/'rul_study'; result=source/'results_merged'
    df=pd.read_csv(source/'data'/'features_and_labels.csv')
    pred_df=pd.read_csv(result/'predictions.csv')
    folds=json.loads((result/'folds.json').read_text(encoding='utf8'))
    if len(folds)!=45: raise ValueError('Expected 15 bearings x 3 seeds')
    summaries=[]; verification=[]
    for f in folds:
        b=f['test_bearing']; seed=int(f['training_seed'])
        ckpt=torch.load(result/'checkpoints'/f'{b}_seed{seed}.pt',map_location='cpu',weights_only=False)
        cols=ckpt['input_columns']; state=ckpt['models']['kan'];
        g=df[df.bearing==b].sort_values('record')
        pd_pred=pred_df[(pred_df.bearing==b)&(pred_df.seed==seed)].sort_values('record')
        assert len(g)==len(pd_pred) and np.array_equal(g.record,pd_pred.record)
        x=(g[cols].to_numpy(dtype=float)-np.asarray(ckpt['scaler_mean']))/np.asarray(ckpt['scaler_scale'])
        X=torch.tensor(x,dtype=torch.float32)
        model=model_cls(len(cols),state['width']);model.load_state_dict(state['state_dict']);model.eval()
        mu=float(state['target_train_mean']); sd=float(state['target_train_std'])
        def predict(t): return torch.clamp(model(t)*sd+mu,min=0)
        with torch.no_grad():
            yhat=predict(X)
            pred0=float(predict(torch.zeros((1,len(cols)))).item())
        max_diff=float(np.max(np.abs(yhat.numpy()-pd_pred['kan_pred_min'].to_numpy())))
        if max_diff>0.002: raise ValueError(f'{b}/{seed} saved prediction mismatch: {max_diff}')
        all_ig=[]
        for begin in range(0,len(X),a.batch_size):
            z=X[begin:begin+a.batch_size]; total=torch.zeros_like(z)
            for alpha in (torch.arange(a.steps,dtype=torch.float32)+.5)/a.steps:
                z_a=(z*alpha).detach().requires_grad_(True)
                predict(z_a).sum().backward()
                total+=z_a.grad.detach()
            all_ig.append(z*total/a.steps)
        ig=torch.cat(all_ig)
        completeness=(ig.sum(dim=1)-(yhat-pred0)).abs()
        mean_abs=ig.abs().mean(dim=0).numpy()
        mean_signed=ig.mean(dim=0).numpy()
        d={'bearing':b,'seed':seed,'condition':g.condition.iloc[0],'records':len(g),
           'prediction_max_abs_diff':max_diff,'completeness_mean_abs_error_minutes':float(completeness.mean()),
           'mean_abs_delta_from_baseline_minutes':float((yhat-pred0).abs().mean()),
           'zero_clipped_fraction':float((yhat==0).float().mean())}
        d.update({f'abs__{c}':float(v) for c,v in zip(cols,mean_abs)})
        d.update({f'signed__{c}':float(v) for c,v in zip(cols,mean_signed)})
        summaries.append(d);verification.append(max_diff)
        print(f'{b} seed {seed}: n={len(g)} max_prediction_diff={max_diff:.6g}',flush=True)
    per_fold=pd.DataFrame(summaries); per_fold.to_csv(a.output/'per_fold_feature_attribution.csv',index=False)
    abs_cols=[f'abs__{c}' for c in cols]
    # Equal-weight each outer bearing and each training seed, never each record.
    per_bearing=per_fold.groupby('bearing')[abs_cols].mean()
    global_imp=per_bearing.mean(axis=0); global_imp.index=cols
    mean_signed=per_fold[[f'signed__{c}' for c in cols]].mean(axis=0);mean_signed.index=cols
    global_summary=pd.DataFrame({'feature':cols,'mean_abs_ig_minutes':global_imp.to_numpy(),'mean_signed_ig_minutes':mean_signed.to_numpy()}).sort_values('mean_abs_ig_minutes',ascending=False)
    global_summary.to_csv(a.output/'global_feature_attribution.csv',index=False)
    pairs=[]
    for b,g in per_fold.groupby('bearing'):
        for (_,a1),(_,a2) in combinations(g.iterrows(),2):
            v1=a1[abs_cols].to_numpy(float);v2=a2[abs_cols].to_numpy(float)
            top1=set(np.argsort(-v1)[:3]);top2=set(np.argsort(-v2)[:3]);
            pairs.append({'bearing':b,'seed_a':int(a1.seed),'seed_b':int(a2.seed),
                          'top3_jaccard':len(top1&top2)/len(top1|top2),
                          'spearman':float(spearmanr(v1,v2).statistic)})
    pairs=pd.DataFrame(pairs);pairs.to_csv(a.output/'within_bearing_seed_stability.csv',index=False)
    across=[]
    for (b1,r1),(b2,r2) in combinations(per_bearing.iterrows(),2):
        i1=set(np.argsort(-r1.to_numpy(float))[:3]);i2=set(np.argsort(-r2.to_numpy(float))[:3]);
        across.append(len(i1&i2)/len(i1|i2))
    per_fold_error=float(per_fold.completeness_mean_abs_error_minutes.mean())
    per_fold_delta=float(per_fold.mean_abs_delta_from_baseline_minutes.mean())
    report={'baseline':'Per fold, bearing-weighted training mean, zero in scaled space; same zero vector for the 13 inputs.',
       'target':'nonnegative clipping applied to predicted minutes to last acquisition',
       'test':'15 held-out bearings x 3 training seeds; no record is an independent bearing',
       'aggregation':'mean absolute IG within each test bearing, then equal mean across 3 seeds and across 15 bearings',
       'ig_steps':a.steps,'max_checkpoint_prediction_error_minutes':max(verification),
       'completeness_mean_absolute_error_minutes':per_fold_error,'mean_abs_prediction_difference_from_baseline_minutes':per_fold_delta,
       'completeness_error_ratio_of_means':per_fold_error/max(per_fold_delta,1e-12),
       'within_bearing_seed_mean_top3_jaccard':float(pairs.top3_jaccard.mean()),
       'within_bearing_seed_median_top3_jaccard':float(pairs.top3_jaccard.median()),
       'within_bearing_seed_mean_spearman':float(pairs.spearman.mean()),
       'between_bearings_mean_top3_jaccard':float(np.mean(across)),
       'between_bearings_note':'Each test bearing uses its own model and training set; not an isolated condition effect.',
       'limits':['Training mean may be physically unrealistic because correlated features/conditions are mixed.',
                 'Speed and radial load are perfectly confounded by three fixed conditions; attribution cannot distinguish independent effects.',
                 'Explains predictions, not physical causality, quality of prediction or future-bearing uncertainty.',
                 '45 models but only 15 independent test bearings; samples in a bearing are correlated.',
                 'Mean signed IG is relative to the training centroid and may cancel across samples; not a monotonicity test.'],
       'global_ranking':global_summary.to_dict(orient='records')}
    (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
