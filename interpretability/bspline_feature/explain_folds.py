"""Cross-fitted held-out integrated gradients for fifteen B-spline KAN folds.

The output is the actual non-negative clipped minute prediction, with the
training feature mean as baseline. Per-record attributions are averaged per
test bearing first, then bearings receive equal weight. Interpretation measures
model reliance, not causality or association with real-world failure physics.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--spline-dir',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path(__file__).parent)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.steps<1: p.error('steps must be positive')
    if a.spline_dir is None: a.spline_dir=a.project/'bspline_study'
    torch.set_num_threads(2)
    sys.path.insert(0,str(a.spline_dir))
    from bspline_layer import BSplineKAN
    study=a.project/'rul_study'
    X=pd.read_csv(study/'data'/'features_and_labels.csv')
    spline=a.spline_dir/f'results_seed{a.seed}'
    folds=json.loads((spline/'folds.json').read_text(encoding='utf8'))
    prediction=pd.read_csv(spline/'predictions.csv')
    if len(folds)!=15 or len(set(f['test_bearing'] for f in folds))!=15:raise ValueError('Require all 15 outer folds')
    if len(prediction)!=9216 or prediction.duplicated(['bearing','record']).any(): raise ValueError('Incomplete or duplicate predictions')
    a.output.mkdir(parents=True,exist_ok=True)
    group=[]; summaries=[]; max_pred_diff=0.; total=0
    for fold in sorted(folds,key=lambda f:f['fold']):
        bearing=fold['test_bearing']
        ck=torch.load(spline/'checkpoints'/f'{bearing}_seed{a.seed}.pt',map_location='cpu',weights_only=False)
        if bearing!=ck['test_bearing'] or bearing in set(ck['train_bearings'])|set(ck['validation_bearings']):raise ValueError('Bearing leakage')
        cols=ck['input_columns']; dim=len(cols)
        g=X[X.bearing==bearing].sort_values('record')
        valid=prediction[prediction.bearing==bearing].sort_values('record')
        if not np.array_equal(g.record,valid.record):raise ValueError(f'Prediction mismatch for {bearing}')
        z=torch.tensor((g[cols].to_numpy(dtype=np.float64)-ck['scaler_mean'])/ck['scaler_scale'],dtype=torch.float32)
        net=BSplineKAN(dim,ck['width'],ck['intervals']);net.load_state_dict(ck['state_dict']);net.eval()
        def predict(q):return (net(q)*ck['target_train_std']+ck['target_train_mean']).clamp(min=0)
        with torch.no_grad():
            yp=predict(z); ref=predict(torch.zeros((1,dim),dtype=torch.float32)).item()
        diff=float(np.max(np.abs(yp.numpy()-valid.predicted_rul_min.to_numpy())))
        max_pred_diff=max(max_pred_diff,diff)
        if diff>0.002:raise ValueError(f'Reconstructed predictions differ for {bearing}: {diff}')
        grad=torch.zeros_like(z)
        for alpha in (torch.arange(a.steps,dtype=torch.float32)+.5)/a.steps:
            x=(alpha*z).detach().requires_grad_(True)
            predict(x).sum().backward();grad+=x.grad.detach()
        ig=z*grad/a.steps
        delta=yp-ref; err=(ig.sum(1)-delta).abs()
        absolute=ig.abs().mean(0).numpy();signed=ig.mean(0).numpy()
        ranked=np.argsort(-absolute)
        nonnegative_fraction=float((yp>0).float().mean());
        group.append(pd.DataFrame({'bearing':bearing,'condition':g.condition.iloc[0],
             'feature':cols,'mean_abs_ig_min':absolute,'mean_signed_ig_min':signed,
             'rank_within_bearing':np.argsort(np.argsort(-absolute))+1,'records':len(g)}))
        summaries.append({'bearing':bearing,'records':len(g),'training_bearings':ck['train_bearings'],
             'validation_bearings':ck['validation_bearings'],'prediction_positive_fraction':nonnegative_fraction,
             'completeness_mean_abs_error_min':float(err.mean()),
             'mean_abs_prediction_minus_baseline_min':float(delta.abs().mean()),
             'ranked_features':[cols[i] for i in ranked],
             'mae_min':float(np.mean(np.abs(g.rul_minutes.to_numpy()-yp.numpy())))})
        total+=len(g)
        print(f'{bearing}: n={len(g)} top3={[cols[i] for i in ranked[:3]]} ig_error={err.mean().item():.5f}',flush=True)
    frame=pd.concat(group,ignore_index=True)
    importance=frame.groupby('feature').agg(mean_abs_ig_min=('mean_abs_ig_min','mean'),
       mean_signed_ig_min=('mean_signed_ig_min','mean'),
       median_rank=('rank_within_bearing','median'),
       top3_bearing_count=('rank_within_bearing',lambda s:int((s<=3).sum())),
       n_bearings=('bearing','nunique')).reset_index().sort_values('mean_abs_ig_min',ascending=False)
    mean_error=np.mean([v['completeness_mean_abs_error_min'] for v in summaries])
    mean_delta=np.mean([v['mean_abs_prediction_minus_baseline_min'] for v in summaries])
    results={'protocol':f'15 fold-specific BSplineKAN models, seed{a.seed}, test bearings only, each model has train-only feature mean baseline. {a.steps} midpoint IG on actual nonnegative clipped RUL output; equal-weight average of per-bearing mean absolute IG.',
       'steps':a.steps,'seed':a.seed,'n_bearings':15,'n_records':total,
       'max_reloaded_prediction_discrepancy_min':max_pred_diff,
       'bearing_equal_completeness_mean_abs_error_min':float(mean_error),
       'bearing_equal_mean_abs_output_difference_from_baseline_min':float(mean_delta),
       'ratio_of_these_equal_bearing_means':float(mean_error/(mean_delta+1e-12)),
       'feature_rankings':importance.to_dict(orient='records'),'per_bearing':summaries,
       'caveats':['Only one training seed. The 15 ranking entries come from 15 distinct trained models and held-out bearings, not 15 independent training trials.',
          'The held-out results are examined for exploratory explanation. Do not select features based on these test results and then report performance on the same bearings as independent evaluation.',
          'RUL target is minutes to last recorded observation, not precise failure time.',
          'Speed and load have only three perfectly linked operating combinations; these attributions cannot separate individual causal effects. Highly correlated vibration features and nonphysical interpolated inputs can make IG baselines and rankings misleading; this is model dependence, not causality.',
          'Absolute attribution is relative to train-mean baseline, not raw-prediction share; the signed and absolute means are different statistics.',
          'Actual predictions are clipped at zero; the clipping kink increases numerical integration error.']}
    (a.output/'metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
    frame.to_csv(a.output/'per_bearing_feature_importance.csv',index=False,encoding='utf-8-sig')
    importance.to_csv(a.output/'global_feature_importance.csv',index=False,encoding='utf-8-sig')
    print('BEARING EQUAL TOP FEATURES:',importance.head(8).to_string(index=False),flush=True)
    print('COMPLETENESS:',mean_error,mean_delta,mean_error/(mean_delta+1e-12),flush=True)

if __name__=='__main__':main()
